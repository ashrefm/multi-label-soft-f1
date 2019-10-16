
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import tensorflow as tf


def micro_bce(y, y_hat):
    """Compute the micro binary cross-entropy on a batch of observations.
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_CLASSES)      
        y_hat (float32 Tensor): prediction output from forward propagation of shape (BATCH_SIZE, N_CLASSES)
        
    Returns:
        cost (scalar Tensor): batch cost value
    """
    # Convert the target array to float32
    y = tf.cast(y, tf.float32)
    # Implement cross entropy loss for each observation and class
    cross_entropy = - y * tf.math.log(tf.maximum(y_hat, 1e-16)) - (1-y) * tf.math.log(tf.maximum(1-y_hat, 1e-16))
    # Average binary cross entropy across all batch observations and classes
    cost = tf.reduce_mean(cross_entropy)
    return cost


def macro_bce(y, y_hat):
    """Compute the macro binary cross-entropy on a batch of observations (average across all classes).
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_CLASSES)      
        y_hat (float32 Tensor): prediction output from forward propagation of shape (BATCH_SIZE, N_CLASSES)
        
    Returns:
        cost (scalar Tensor): batch cost value
    """
    # Convert the target array to float32
    y = tf.cast(y, tf.float32)
    # Implement cross entropy loss for each observation and class
    cross_entropy = - y * tf.math.log(tf.maximum(y_hat, 1e-16)) - (1-y) * tf.math.log(tf.maximum(1-y_hat, 1e-16))
    # Average all binary cross entropy losses over the whole batch for each class
    cost = tf.reduce_mean(cross_entropy, axis=0)
    # Average all binary cross entropy losses over classes within the batch
    cost = tf.reduce_mean(cost)
    return cost


def macro_soft_f1_loss(y, y_hat):
    """Compute the macro soft f1-score (average soft-f1 score across all classes).
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_CLASSES)
        y_hat (float32 Tensor): prediction output from forward propagation of shape (BATCH_SIZE, N_CLASSES)
        
    Returns:
        cost (scalar Tensor): batch cost value
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    TP = tf.reduce_sum(y_hat * y, axis=0)
    FP = tf.reduce_sum(y_hat * (1 - y), axis=0)
    FN = tf.reduce_sum((1 - y_hat) * y, axis=0)
    precision = TP / (TP + FP + 1e-16)
    recall = TP / (TP + FN + 1e-16)
    soft_f1 = 2 * precision * recall / (precision + recall + 1e-16)
    cost = 1 - soft_f1 # reduce 1-f1 in order to increase f1
    macro_cost = tf.reduce_mean(cost)
    return macro_cost


def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1 score on a batch of observations (average F1 across classes)
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_CLASSES)
        y_hat (float32 Tensor): prediction output from forward propagation of shape (BATCH_SIZE, N_CLASSES)
        thresh: probability value beyond which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 in batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    TP = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    FP = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    FN = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    precision = TP / (TP + FP + 1e-16)
    recall = TP / (TP + FN + 1e-16)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1


def learning_curves(history):
    """Plot the learning curves of loss and macro f1 score 
    for the training and validation datasets.
    
    Args:
        history: history callback of fitting a tensorflow keras model 
    """
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    macro_f1 = history.history['macro_f1']
    val_macro_f1 = history.history['val_macro_f1']

    style.use("bmh")
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    plt.subplot(2, 1, 2)
    plt.plot(macro_f1, label='Training Macro-F1')
    plt.plot(val_macro_f1, label='Validation Macro-F1')
    plt.legend(loc='lower right')
    plt.ylabel('Macro-F1')
    plt.title('Training and Validation Macro-F1')
    plt.xlabel('epoch')

    plt.show()
    
    return loss, val_loss, macro_f1, val_macro_f1


def perf_grid(ds, target, class_names, model, n_thresh=100):
    """Computes the performance table containing target, label names,
    thresholds between 0 and 1, number of tp, fp, fn, precision,
    recall and f-score metrics for each class.
    
    Args:
        ds (tf.data.Datatset): contains the features array
        target (numpy array): target matrix of shape
        class_names (list of strings): column names in target matrix
        model (tensorflow keras model): model to use for prediction
        n_thresh (int) : number of thresholds to try
        
    Returns:
        grid (Pandas dataframe): performance table 
    """
    
    # Get predictions
    y_hat_val = model.predict(ds)
    # Define target matrix
    y_val = target
    # Find class frequencies in the validation set
    class_freq = target.sum(axis=0)
    # Get class indexes
    classes = [i for i in range(len(class_names))]
    # Define thresholds
    thresholds = np.linspace(0,1,n_thresh+1).astype(np.float32)
    
    # Compute all metrics for all classes
    targets, labels, tps, fps, fns, precisions, recalls, f1s = [], [], [], [], [], [], [], []
    for c in classes:
        for thresh in thresholds:   
            targets.append(c)
            labels.append(class_names[c])
            y_hat = y_hat_val[:,c]
            y = y_val[:,c]
            y_pred = y_hat > thresh
            tp = np.count_nonzero(y_pred  * y)
            fp = np.count_nonzero(y_pred * (1-y))
            fn = np.count_nonzero((1-y_pred) * y)
            precision = tp/(tp+fp+1e-16) if (tp+fp)!=0 else 0
            recall = tp/(tp+fn+1e-16) if (tp+fn)!=0 else 0
            f1 = 2*precision*recall/(precision+recall+1e-16)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            
    # Create the performance dataframe
    grid = pd.DataFrame({
        'target':targets,
        'label':labels,
        'threshold':list(thresholds)*len(classes),
        'tp':tps,
        'fp':fps,
        'fn':fns,
        'precision':precisions,
        'recall':recalls,
        'f1':f1s})
    
    return grid