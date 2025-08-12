import tensorflow as tf
from keras.utils import plot_model
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import numpy as np
import os
from Sub_Functions.Evaluate import main_est_parameters
from keras.optimizers import Adam

from tensorflow.keras import layers, models


import tensorflow as tf
from tensorflow.keras import layers

class StructuralAttention(layers.Layer):
    def __init__(self, num_heads=4, key_dim=64, mlp_ratio=4, local_window=3, threshold=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mlp_ratio = mlp_ratio
        self.local_window = local_window
        self.threshold = threshold

        # Patch classifier for FG/BG
        self.patch_classifier = layers.Conv2D(1, kernel_size=1, activation='sigmoid')

        # Normalization layers
        self.norm_fg = layers.LayerNormalization(epsilon=1e-6)
        self.norm_bg = layers.LayerNormalization(epsilon=1e-6)

        # Attention layers
        self.attn_fg = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.attn_bg = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

        # MLP for FG and BG
        self.mlp_fg = tf.keras.Sequential([
            layers.Dense(key_dim * mlp_ratio, activation='gelu'),
            layers.Dense(key_dim)
        ])
        self.mlp_bg = tf.keras.Sequential([
            layers.Dense(key_dim * mlp_ratio, activation='gelu'),
            layers.Dense(key_dim)
        ])

    def call(self, x):
        """
        x: [B, H, W, C] feature map
        """
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # 1. Predict FG/BG mask
        mask = self.patch_classifier(x)  # [B,H,W,1]
        mask_flat = tf.reshape(mask, [B, H * W])  # [B, HW]

        # Flatten feature map
        tokens = tf.reshape(x, [B, H * W, C])  # [B, HW, C]

        # 2. Foreground attention (structural attention)
        fg_mask = tf.cast(mask_flat > self.threshold, tf.float32)
        fg_tokens = tokens * tf.expand_dims(fg_mask, -1)  # keep only FG tokens
        fg_out = self.attn_fg(self.norm_fg(fg_tokens), self.norm_fg(fg_tokens))
        fg_out = fg_out + fg_tokens  # residual
        fg_out = self.mlp_fg(self.norm_fg(fg_out)) + fg_out

        # 3. Background attention (local window)
        bg_mask = 1.0 - fg_mask
        bg_tokens = tokens * tf.expand_dims(bg_mask, -1)
        bg_map = tf.reshape(bg_tokens, [B, H, W, C])
        bg_local = layers.DepthwiseConv2D(self.local_window, padding='same')(bg_map)
        bg_local = tf.reshape(bg_local, [B, H * W, C])
        bg_out = self.attn_bg(self.norm_bg(bg_local), self.norm_bg(bg_local))
        bg_out = bg_out + bg_local
        bg_out = self.mlp_bg(self.norm_bg(bg_out)) + bg_out

        # 4. Merge FG + BG outputs
        merged_tokens = fg_out * tf.expand_dims(fg_mask, -1) + \
                        bg_out * tf.expand_dims(bg_mask, -1)

        # Reshape back to [B, H, W, C]
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

    # Channel Attention
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

    # Spatial Attention
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


# --- Hybrid Structural Attention Enabled Explainable CNN ---


def blocks(input_shape, num_classes, out_channels=64, window_size=7, threshold=0.5):
    """
    Define the hybrid model with Structural Attention integrated.
    """
    inputs = Input(shape=input_shape)

    # Initial Convolution Block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Apply Structural Attention after the first convolutional block

    x = StructuralAttention(num_heads=2, key_dim=32, threshold=0.5)(x)
    # Apply CBAM Attention Block
    x = cbam_block(x)  # Attention block

    # Continue with additional layers after attention block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)

    # Define the model
    model = models.Model(inputs=inputs, outputs=[outputs])
    return model




# --- Proposed Function to Train and Evaluate ---
def proposed_model(x_train, x_test, y_train, y_test, train_percent, DB):
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    model = blocks(input_shape, num_classes)

    # model.compile(optimizer='adam',
    #               loss={'output': 'categorical_crossentropy', 'attention_map': lambda y_true, y_pred: 0.0},
    #               metrics={'output': 'accuracy'})

    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    os.makedirs("Architectures/", exist_ok=True)

    # model.fit(x_train, {'output': y_train_cat, 'attention_map': np.zeros((x_train.shape[0], 25, 25, 1))},
    #           validation_data=(x_test, {'output': y_test_cat, 'attention_map': np.zeros((x_test.shape[0], 25, 25, 1))}),
    #           epochs=epochs,
    #           verbose=2)

    # y_pred_probs, _ = model.predict(x_test)
    # y_pred = np.argmax(y_pred_probs, axis=1)
    #
    # print("\nClassification Report:\n")
    # print(classification_report(y_test, y_pred))
    #
    # metrics = {
    #     'accuracy': np.mean(y_pred == y_test),
    #     'classification_report': classification_report(y_test, y_pred, output_dict=True)
    # }
    #
    # return metrics

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
            model.fit(x_train, y_train_cat, epochs=end_epochs, initial_epoch=prev_epoch, batch_size=8,
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

