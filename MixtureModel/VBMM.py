# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans
from scipy.special import psi as digamma


class VBMM(object):
    def __init__(self, input, max_components=20, b=1e6, c=1e-5, lambda0=5):
        
        self.m = max_components
        self.b = b 
        self.c = c
        self.lambda0 = lambda0
        self.input = input
        self.N = input.shape[0]
        
        kmeans = self.InitializeEM(input)
        
        self.MuS = np.reshape(kmeans.cluster_centers_, (self.m,))
        self.mS = self.MuS
        self.vS = np.asarray([self.var_c_norm]).flatten()

        self.m0 = np.mean(input)
        self.v0 = ( ( np.max(input) - np.min(input) ) / 3 )**2

        self.bS = np.asarray([1.] * self.m)
        self.cS = self.vS
        self.LambdaS = self.Pi * 100
        self.gamma_n_s = None
        
        self.means = None
        self.sigmas = None
        self.pis = None
        self.components = None
        self.number_of_components = None
        
        
    def InitializeEM(self, input):
        kmeans = KMeans(init='k-means++', n_clusters=self.m, n_init=3)
        kmeans.fit(input[:,np.newaxis])
        kmeans.labels_ = kmeans.labels_ + 1
        
        self.Pi = np.asarray([np.count_nonzero(kmeans.labels_[kmeans.labels_ == i+1]) / np.float(self.N) for i in range(self.m)])
    
        var_classes = []
        for c in range(self.m):
            sum_c = np.array([0.0])
            for i in range(self.N):
                if kmeans.labels_[i] == c+1:
                    sum_c = sum_c + (self.input[i] - kmeans.cluster_centers_[c])**2
            
            var_classes.append(sum_c)
        
        var_c_norm = [ var_classes[i] / (self.Pi[i]*self.N)  for i in range(len(var_classes))]
        var_c_norm = [ var_c_norm[i][0] / (self.Pi[i]*self.N)  if var_c_norm[i][0] / (self.Pi[i]*self.N) > 0.01 else 0.01  for i in range(len(var_c_norm))]
        self.var_c_norm = var_c_norm

        return kmeans
        
        
    def Estep(self):
        
        log_pi_s_hat = np.asarray([digamma(self.LambdaS[i]) - digamma(np.sum(self.LambdaS)) for i in range(self.m)])
        log_beta_s_hat = digamma(self.cS) - np.log(self.bS)
        beta_s_mean = self.bS * self.cS
        temp = np.asarray([(-1/2.) * beta_s_mean * (self.input[i]**2 + self.mS**2 + self.vS**2 - 2*self.mS*self.input[i]) for i in range(self.N)]).T    
    
        gamma_n_s_hat = np.exp(temp.T) * np.exp(log_pi_s_hat) * (np.exp(log_beta_s_hat))**(1/2.)
        gamma_n_s_hat[gamma_n_s_hat <=0] = np.finfo(np.float).eps
        gamma_n_s = gamma_n_s_hat/gamma_n_s_hat.sum(axis=1)[:,None] 
        gamma_n_s[gamma_n_s <=0] = np.finfo(np.float).eps
    
        return gamma_n_s
        
        
    def Mstep(self):
    
        pi_s_mean = (np.sum(self.gamma_n_s, axis=0)) / self.N
        N_s_mean = self.N * pi_s_mean
        y_s_mean = np.asarray([np.sum(self.gamma_n_s[:,i] * self.input) / self.N for i in range(self.m)])
        y_s_square_mean = np.asarray([np.sum(self.gamma_n_s[:,i] * self.input**2) / self.N for i in range(self.m)])
        sigma_s_square_mean = y_s_square_mean + pi_s_mean * (self.mS**2 + self.vS) - 2*self.mS*y_s_mean
    
        LambdaS_ = N_s_mean + self.lambda0
        one_div_bS = (self.N * sigma_s_square_mean) / 2 + (1/self.b)
        bS_ = 1 / one_div_bS
        cS_ = (N_s_mean / 2) + self.c
        m_data_s = y_s_mean / pi_s_mean
        tau_data_s = N_s_mean * bS_ * cS_
        tau_s = (1 / self.v0) + tau_data_s
        mS_ = self.m0/(self.v0*tau_s) + (tau_data_s * m_data_s) / tau_s
        vS_ = (1/bS_) / (self.N*pi_s_mean)
    
        return LambdaS_, bS_, cS_, mS_, vS_ 
        
        
    def Fit(self, crit = 1e-6, compress=False):
        
        converged = False
        counter = 0

        while not converged:      
            print ("Current interation {}  ".format(counter))
            if counter>200:
                print "Break...."
                break
            
            gamma_n_s = self.Estep()
            self.gamma_n_s = gamma_n_s
            
            Lambda_s_new, bS_new, cS_new, mS_new, vS_new = self.Mstep()            
            converged = (np.all(np.abs((np.array(mS_new) - np.array(self.mS)) < crit)))           
            self.LambdaS, self.bS, self.cS, self.mS, self.vS = Lambda_s_new, bS_new, cS_new, mS_new, vS_new        
            counter = counter + 1
    
        sigma_est = np.sqrt(self.vS)
        components = np.sum(gamma_n_s, axis=0)/self.N > 5e-2
        pi_est = np.sum(self.gamma_n_s, axis=0)/self.N 
        number_of_components = np.sum(components)

        if compress==False:
            print ("The algorithm converged in {} iterations and found {} component(s)".format(counter, number_of_components))
            for i in range(number_of_components):
                print("Component %d: mean: %4f - std: %4f, pi: %4f" % (i+1, self.mS[components][i], sigma_est[components][i], pi_est[components][i]))
            
        normalize_factor = 1. / np.sum(pi_est[components])            
            
            
        self.means = self.mS
        self.sigmas = sigma_est
        self.pis = pi_est * normalize_factor
        self.components = components
        self.number_of_components = number_of_components