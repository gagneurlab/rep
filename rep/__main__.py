import os
# import drmaa
# from multiprocessing import Process, Queue, Pool
from joblib import Parallel, delayed
import queue
import time
import math
import _pickle as cPickle

# from data.gtex import GTEx
# from data.expecto import Expecto

from rep import preprocessing as p
from rep.models.linear_regression import Linear_Regression
from rep.models.linear_regression import Transform
from rep.models.linear_regression import FeatureReduction


# from cli.train import train_keras
# from data import create_lmdb

def wc_l(fname):
    """Get the number of lines of a text-file using unix `wc -l`
    """
    import subprocess
    return int((subprocess.Popen('wc -l {0}'.format(fname), shell=True, stdout=subprocess.PIPE).stdout).readlines()[0].split()[0])


def cross(norm_gtex):
    # load invidivudals
    path = os.path.join(os.readlink(os.path.join("..","data")),"processed","gtex","recount")

    train = []
    valid = []
    test = []
    states = ['train','valid','test']
    dict_states_indiv = {'train':train,'valid':valid,'test':test}

    for s in states:
        with open(os.path.join(path,s+"_individuals.txt"), 'r') as f:
            for l in f: dict_states_indiv[s].append(l.replace("\n",""))


    selected_genes = norm_gtex.obs_names # training with all features
    print("Total Genes: ",len(selected_genes))

    # compute cross tissue matrix
#     (X_train_norm, Y_train_norm, samples_description_train, gene_id_train) = p.rnaseq_cross_tissue(norm_gtex, individuals=train, gene_ids=selected_genes, onlyBlood = True)
    (X_valid_norm, Y_valid_norm, samples_description_valid, gene_id_valid) = p.rnaseq_cross_tissue(norm_gtex, individuals=valid, gene_ids=selected_genes, onlyBlood = True)
    (X_test_norm, Y_test_norm, samples_description_test, gene_id_test) = p.rnaseq_cross_tissue(norm_gtex, individuals=test, gene_ids=selected_genes, onlyBlood = True)
    
#     return  (X_train_norm, Y_train_norm, samples_description_train, gene_id_train), (X_valid_norm, Y_valid_norm, samples_description_valid, gene_id_valid), (X_test_norm, Y_test_norm, samples_description_test, gene_id_test)
    return (X_valid_norm, Y_valid_norm, samples_description_valid, gene_id_valid), (X_test_norm, Y_test_norm, samples_description_test, gene_id_test)

def run_batch(q, Xs_reduced, Ys, Xs_valid_reduced, Ys_valid, batch_size, start):
    
    cpu = q.get()
    print (start, cpu)

    # Put here your job cmd
    stop = min(start+batch_size,Ys_valid.shape[1])

    for i in range(start,stop):
        arr_output = []
        m = Linear_Regression(Xs_reduced, Ys[:,i], Xs_valid_reduced, Ys_valid[:,i])
        arr_output.append({i: m.run('train_lassolars_model')})
    
    pickle_out = open("dict.pickle" + str(start),"wb")
    cPickle.dump(arr_output, pickle_out,protocol=2)
     

    # return cpu id to queue
    q.put(cpu)
    
        
def main_parallel(Xs_reduced, Ys, Xs_valid_reduced, Ys_valid):
    
    
#     jobs = [20, 25, 30]
#     batches = [80, 90, 100]
    
    batches = [100]
    
    N_CPU = 10
    
    q = queue.Queue(maxsize=N_CPU)
    for i in range(N_CPU):
        q.put(i)


    for k in batches:
        print("Test number jobs ", N_CPU, " batch ", k)
        count_batches = int(math.floor(Ys_valid.shape[1]/k))
        print("Baches count: ", count_batches)            
        tic = time.time()
        Parallel(n_jobs=N_CPU,backend="threading")(delayed(run_batch)(q, Xs_reduced, Ys, Xs_valid_reduced, Ys_valid, k, i*k) for i in range(count_batches))
        
        toc = time.time()
        print('\nElapsed time computing the average of couple of slices {:.2f} s'.format(toc - tic))

def main():
    # assembling:
    # parser = argh.ArghParser()
    # parser.add_commands([wc_l, train_keras, create_lmdb])
    # parser.add_commands([wc_l])
    # argh.dispatch(parser)

    # gtex = GTEx()
    #
    # print("1. Load GTEx data:")
    # gtex.load_count_matrix("../data.csv", sep=",", varanno="../anno.csv", obsanno="../anno_obs.csv")
    # gtex.get_count_matrix()
    #
    #
    # print("2. Filter data:")
    
    
    
    # load gtex data and compute the cross tissue
    file = os.path.join("..","..","recount_gtex_logratios.h5ad")
    print("Loading file: ", file)
    gtex = p.load(file)
    
    (X_valid, Y_valid, _, _), (X_test, Y_test, _, _) = cross(gtex)
#     (X_train, Y_train, _, _), (X_valid, Y_valid, _, _), (X_test, Y_test, _, _) = cross(gtex)
    
    # normalize and dimensionality reduction
    (Xs,Ys,x_scaler,y_scaler) = Transform(X_valid,Y_valid).fit_transform()
    (Xs_valid, Ys_valid) = Transform(X_test,Y_test).transform(x_scaler, y_scaler)
    (Xs_reduced,pca) = FeatureReduction(Xs).pca_svd(components = 1000,fit_transform = True)
    (Xs_valid_reduced,_) = FeatureReduction(Xs_valid).pca_svd(components = 1000,fit_transform = False, scaler = pca)

    main_parallel(Xs_reduced, Ys, Xs_valid_reduced, Ys_valid)
    
#     pred = np.zeros(Ys_valid.shape)
#     procs = []
    
#     # run linear regression
#     for i in range(190):
#         proc = Process(target=run_batch, args=(Xs_reduced, Ys, Xs_valid_reduced, Ys_valid, pred, i*100, 100))
#         procs.append(proc)
#         proc.start()
    
#     for proc in procs:
#         proc.join()
    
    #     with drmaa.Session() as s:
    #         print('A session was started successfully')
    #         jt = s.createJobTemplate()

    # #         for i in range(Ys_valid.shape/100):
    # #             jobid = s.runJob(run_batch(Xs_reduced, Ys, Xs_valid_reduced, Ys_valid, pred, start=100*i, batch_size = 100*(i+1)))
    # #         print('Your job has been submitted with ID %s' % jobid)

    #         retval = s.wait(jobid, drmaa.Session.TIMEOUT_WAIT_FOREVER)
    #         print('Job: {0} finished with status {1}'.format(retval.jobId, retval.hasExited))


    #         print('Cleaning up')
    #         s.deleteJobTemplate(jt)
    
#     print(pred[:3])

if __name__ == "__main__":
    
    start = time.time()
    main()
    end = time.time()
    print("Ellapsed time: ", end - start)

