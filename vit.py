import tensorflow as tf
import tensorflow_addons as tfa
from keras import layers
from keras.layers import Dense, Dropout, LayerNormalization
from keras.layers import Rescaling

from Sub_Functions.Evaluate import main_est_parameters


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class StructuralAttention(layers.Layer):
    def __init__(self, num_heads=4, key_dim=64, mlp_ratio=4, local_window=3, threshold=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mlp_ratio = mlp_ratio
        self.local_window = local_window
        self.threshold = threshold

        self.patch_classifier = layers.Conv2D(1, kernel_size=1, activation='sigmoid')

        self.norm_fg = layers.LayerNormalization(epsilon=1e-6)
        self.norm_bg = layers.LayerNormalization(epsilon=1e-6)

        self.attn_fg = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.attn_bg = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

        self.mlp_fg = tf.keras.Sequential([
            layers.Dense(key_dim * mlp_ratio, activation='gelu'),
            layers.Dense(key_dim)
        ])
        self.mlp_bg = tf.keras.Sequential([
            layers.Dense(key_dim * mlp_ratio, activation='gelu'),
            layers.Dense(key_dim)
        ])

    def call(self, x):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        mask = self.patch_classifier(x)  # [B,H,W,1]
        mask_flat = tf.reshape(mask, [B, H * W])  # [B, HW]
        tokens = tf.reshape(x, [B, H * W, C])  # [B, HW, C]

        fg_mask = tf.cast(mask_flat > self.threshold, tf.float32)
        fg_tokens = tokens * tf.expand_dims(fg_mask, -1)
        fg_out = self.attn_fg(self.norm_fg(fg_tokens), self.norm_fg(fg_tokens))
        fg_out = fg_out + fg_tokens
        fg_out = self.mlp_fg(self.norm_fg(fg_out)) + fg_out

        bg_mask = 1.0 - fg_mask
        bg_tokens = tokens * tf.expand_dims(bg_mask, -1)
        bg_map = tf.reshape(bg_tokens, [B, H, W, C])
        bg_local = layers.DepthwiseConv2D(self.local_window, padding='same')(bg_map)
        bg_local = tf.reshape(bg_local, [B, H * W, C])
        bg_out = self.attn_bg(self.norm_bg(bg_local), self.norm_bg(bg_local))
        bg_out = bg_out + bg_local
        bg_out = self.mlp_bg(self.norm_bg(bg_out)) + bg_out

        merged_tokens = fg_out * tf.expand_dims(fg_mask, -1) + bg_out * tf.expand_dims(bg_mask, -1)
        out = tf.reshape(merged_tokens, [B, H, W, C])
        return out


def cbam_block(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_layer_one = layers.Dense(channel // ratio,
                                    activation='relu',
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros')
    shared_layer_two = layers.Dense(channel,
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros')

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)
    channel_refined_feature = layers.Multiply()([input_feature, cbam_feature])

    avg_pool = tf.reduce_mean(channel_refined_feature, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(channel_refined_feature, axis=-1, keepdims=True)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    spatial_attention = layers.Conv2D(filters=1,
                                      kernel_size=7,
                                      strides=1,
                                      padding='same',
                                      activation='sigmoid',
                                      kernel_initializer='he_normal',
                                      use_bias=False)(concat)
    refined_feature = layers.Multiply()([channel_refined_feature, spatial_attention])
    return refined_feature


class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        image_size,
        patch_size,
        num_layers,
        num_classes,
        d_model,
        num_heads,
        mlp_dim,
        channels=5,  # Changed to 5 channels
        dropout=0.1,
    ):
        super(VisionTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.rescale = Rescaling(1. / 255)
        self.pos_emb = self.add_weight(
            "pos_emb", shape=(1, num_patches + 1, d_model)
        )
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, d_model))
        self.patch_proj = Dense(d_model)
        self.structural_attention = StructuralAttention(num_heads=4, key_dim=d_model // 4)
        # d_model//4 because StructuralAttention's key_dim is 64 default; adjust as needed

        self.enc_layers = [
            TransformerBlock(d_model, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ]
        self.mlp_head = tf.keras.Sequential(
            [
                Dense(mlp_dim, activation=tfa.activations.gelu),
                Dropout(dropout),
                Dense(num_classes),
            ]
        )

    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        x = self.rescale(x)  # Rescale pixels to [0,1]

        # Extract patches
        patches = self.extract_patches(x)

        # Project patches to embedding dimension
        x = self.patch_proj(patches)  # shape: [B, num_patches, d_model]

        # Reshape for Spatial Attention: convert to [B, H, W, C]
        # num_patches = (image_size//patch_size) ** 2
        # so H=W=num_patches^(1/2)
        num_patches = tf.shape(x)[1]
        spatial_dim = tf.cast(tf.math.sqrt(tf.cast(num_patches, tf.float32)), tf.int32)
        x_reshape = tf.reshape(x, [batch_size, spatial_dim, spatial_dim, self.d_model])

        # Apply Structural Attention
        x_reshape = self.structural_attention(x_reshape)

        # Apply CBAM attention
        x_reshape = cbam_block(x_reshape)

        # Flatten back to sequence shape [B, num_patches, d_model]
        x = tf.reshape(x_reshape, [batch_size, num_patches, self.d_model])

        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb

        for layer in self.enc_layers:
            x = layer(x, training)

        x = self.mlp_head(x[:, 0])
        return x

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

def train_vision_transformer(x_train, x_test, y_train, y_test, train_percent, DB,
                             image_size, patch_size, num_layers, num_classes, d_model, num_heads, mlp_dim, dropout=0.1,
                             batch_size=8):
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Instantiate the VisionTransformer from the integrated code (assumed imported or defined)
    model = VisionTransformer(image_size=image_size,
                              patch_size=patch_size,
                              num_layers=num_layers,
                              num_classes=num_classes,
                              d_model=d_model,
                              num_heads=num_heads,
                              mlp_dim=mlp_dim,
                              channels=input_shape[-1],  # should be 5 channels as per discussion
                              dropout=dropout)

    # Compile model with Adam optimizer and categorical crossentropy loss
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    os.makedirs("Architectures/", exist_ok=True)

    base_epochs = [1, 200, 300, 400, 500]
    total_epochs = base_epochs[-1]

    Checkpoint_dir = f"Checkpoint/{DB}/TP_{int(train_percent * 100)}"
    os.makedirs(Checkpoint_dir, exist_ok=True)
    metric_path = f"Analysis/Performance_Analysis/With_Smote/{DB}/"
    os.makedirs(metric_path, exist_ok=True)
    prev_epoch = 0

    for ep in reversed(base_epochs):
        ckt_path = os.path.join(Checkpoint_dir, f"model_epoch_{ep}.weights.h5")
        metrics_path = os.path.join(metric_path, f"metrics_{train_percent}percent_epoch{ep}.npy")
        if os.path.exists(ckt_path) and os.path.exists(metrics_path):
            print(f"Found existing full checkpoint and metrics for epoch {ep}, loading and resuming...")
            model.load_weights(ckt_path)
            prev_epoch = ep
            break

    metrics_all = {}
    for end_epochs in base_epochs:
        if end_epochs <= prev_epoch:
            continue

        print(f" Training from epoch {prev_epoch + 1} to {end_epochs} for TP={train_percent}%...")

        ckt_path = os.path.join(Checkpoint_dir, f"model_epoch_{end_epochs}.weights.h5")
        metrics_path = os.path.join(metric_path, f"metrics_{train_percent}percent_epoch{end_epochs}.npy")

        try:
            model.fit(x_train, y_train_cat, epochs=end_epochs, initial_epoch=prev_epoch, batch_size=batch_size,
                      validation_split=0.2)
            plot_model(model, to_file=f"Architectures/model_architecture.png", show_shapes=True, show_layer_names=True)
            model.save(f"Saved_model/{DB}_model.h5")
            model.save_weights(ckt_path)
            print(f"Checkpoint saved at: {ckt_path}")

            preds = model.predict(x_test)
            y_pred2 = np.argmax(preds, axis=1)
            y_test_labels = np.argmax(y_test_cat, axis=1)
            metrics = main_est_parameters(y_test_labels, y_pred2)

            metrics_all[f"epoch_{end_epochs}"] = metrics
            np.save(metrics_path, metrics)
            print(f"Metrics saved at: {metrics_path}")
            prev_epoch = end_epochs
            model.save(f"Saved_model/{DB}_model.h5")
            model.save(f"Saved_model/{DB}_model.keras")
        except KeyboardInterrupt:
            print(
                f"Training interrupted during epoch chunk {prev_epoch + 1}-{end_epochs}. Not saving checkpoint or metrics.")
            raise
        print(f"\nCompleted training for {train_percent}% up to {prev_epoch} epochs.")
    return metrics_all



# This code snippet assumes:
#
# You have your integrated Vision Transformer model available with signature:
#
# python
# VisionTransformer(image_size, patch_size, num_layers, num_classes, d_model, num_heads, mlp_dim, channels, dropout)
# The main_est_parameters function is defined for evaluation metrics as in your original usage.
#
# The structure and functionality for restoring checkpoints, saving metrics, and saving models are preserved.
#
# You can now call train_vision_transformer(...) with appropriate parameters and data to train the transformer model in the same training loop style as your provided code.