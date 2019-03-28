import tensorflow as tf
from model import Autoencoder
import time
import numpy as np
def add_noise(imgs, db=10, per_image=False):
    amp = 10**(-db/10)
    noise = np.random.normal(scale=amp, size=imgs.shape)
    if per_image:
        noise *= np.random.uniform(0,1, size=noise.shape[0])[:, np.newaxis, np.newaxis, np.newaxis]
    imgs += noise
    imgs += np.amin(imgs, axis=(1,2), keepdims=True)
    imgs /= np.amax(imgs, axis=(1,2), keepdims=True)
    return imgs

def trainer(model_object, input_tensor, label_tensor, train_size, datasess, test_data, use_conv=False, loss_type='CE', learning_rate=1e-4, alpha_r=3., 
            batch_size=128, num_epoch=10, n_z=32, log_step=5, with_noise=False, val_step=1, bench_model=None, early_stop=10, stop_tol=0.05):

    model = model_object(sess=None, input_tensor=None, loss_type=loss_type, use_conv=use_conv, alpha_r=alpha_r,
        learning_rate=learning_rate, batch_size=batch_size, n_z=n_z)
    
    validation_hist = []
    best = None
    label_replace = np.arange(343)
    label_replace[134] = 135

    for epoch in range(num_epoch):
        start_time = time.time()
        for iter in range(train_size // batch_size):
            # Get a batch
            #batch = mnist.train.next_batch(batch_size)
            batch = datasess.run([input_tensor, label_tensor])
            
            if with_noise:
                batch[0] = add_noise(batch[0], per_image=True)
            # Execute the forward and backward pass 
            # Report computed losses
            batch_lab = label_replace[batch[1]]
            losses = model.run_single_step(batch[0], batch_lab)#batch[0])
        end_time = time.time()
        
        if epoch % log_step == 0:
            log_str = '[Epoch {}] '.format(epoch)
            for k, v in losses.items():
                print(k, v)
                log_str += '{}: {:.3f}  '.format(k, v)
            log_str += '({:.3f} sec/epoch)'.format(end_time - start_time)
            print(log_str)
            
        if epoch % val_step == 0 and bench_model is not None:
            val_loss = validate(model, bench_model, test_data)
            validation_hist.append(val_loss)
            if best is None:
                best = (0, val_loss)
            elif best[1]*(1 - stop_tol) > val_loss:
                best = (epoch, val_loss)
            elif best[0] + early_stop < epoch:
                print("Stopping Early")
                break
            print("Validation Loss: ", val_loss)
            
    print('Done!')
    return model

def evaluate(zz, zz_, eval_size=1000):
    """
        zz: from model to be evaluated
        zz_: bench mark model
    """
    ztest = zz[:eval_size]
    zdist = np.sum(ztest[:, np.newaxis, :]* zz[np.newaxis, ...], axis=-1)
    ztest_ = zz_[:eval_size]
    zdist_ = np.sum(ztest_[:, np.newaxis, :]* zz_[np.newaxis, ...], axis=-1)
    
    zrank = np.argsort(zdist, axis=-1)[:,::-1]   #ranking by model
    zrank_ = np.argsort(zdist_, axis=-1)[:,::-1]   #ranking by benchmark
    comp_dist = np.asarray([zdist_[i][zrank[i]] for i in range(eval_size)])
    comp_dist_ = np.asarray([zdist_[i][zrank_[i]] for i in range(eval_size)])
    return comp_dist, comp_dist_
    
    
def validate(model, model_, data, db=10, cutoff=200):
    zz_ = model_.transformer(data)
    data = add_noise(data.copy(), db)
    zz = model.transformer(data)
    
    comp_dist, comp_dist_ = evaluate(zz, zz_, eval_size=1000)
    comp_dist = comp_dist[:,:cutoff]
    comp_dist_ = comp_dist_[:,:cutoff]
    loss = np.mean(np.sum((comp_dist - comp_dist_)**2, axis=1), axis=0) 
    return loss
