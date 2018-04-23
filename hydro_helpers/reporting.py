import numpy as np
import subprocess

def cv_report(results, n_top=3, score='NSE'):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_'+score] == i)
        for candidate in candidates:
            mean = results['mean_test_'+score][candidate]
            mean = -mean if score=='RMSE' else mean
            std = results['std_test_'+score][candidate]
            print('\n', '+'*70, '\n')
            print("Model with rank: {0}".format(i))
            print("Mean validation score (",score,"): {0:.3f} (std: {1:.3f})".format(mean,std))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("\n", " Multi-Metric CV Matrix:", "\n")
            for eff in ['NSE', 'KGE', 'RMSE']:
                score_array = []
                j = 0
                split = 'split'+'0'+'_test_'+eff
                while split in results.keys():
                    j+=1
                    grade = -results[split][candidate] if eff=='RMSE' else results[split][candidate]
                    score_array.append(grade)
                    split = 'split'+str(j)+'_test_'+eff
                    
                print("  CV {0} with splits 0~{1}: ".format(eff, j-1),
                    " | ".join('[{0}]:{1:.3f}'.format(*s) for s in enumerate(score_array)))
    print('\n', '+'*70)

    return None
