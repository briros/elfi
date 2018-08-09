import elfi
from elfi.examples import ma2
import matplotlib.pyplot as plt


# load the model from elfi.examples
model = ma2.get_model(n_obs=100, true_params=[-0.5,0.1])

# setup and run rejection sampling
bolfi = elfi.BOLFI(model['d'], batch_size=1, initial_evidence=20,
    update_interval=5, bounds={'t1':(-2,2),'t2':(-1,1)},
    acq_noise_var=[0.1,0.1])
result = bolfi.fit(n_evidence=300)

result_BOLFI = bolfi.sample(n_samples = 10000, algorithm = 'smc')
samples = np.append(result_BOLFI.samples['t1'].reshape(-1,1),
                    result_BOLFI.samples['t2'].reshape(-1,1),axis=1)

plt.hist2d(samples[:,0],samples[:,1],(50,50),cmap=plt.cm.jet)
plt.show()
    #print(samples)


#samples = np.append(result_BOLFI.samples['t1'].reshape(-1,1),
#                    result_BOLFI.samples['t2'].reshape(-1,1),axis=1)
# plt.plot(result_BOLFI.samples['t1'].reshape(-1,1))
# plt.show()

# sample(self,
#            n_samples=,
#            warmup=None,
#            n_chains=4,
#           threshold=None,
#           initials=None,
#           algorithm='nuts',
#           n_evidence=None,
#           **kwargs):

# Minimizer
# print(bolfi.extract_result().x_min)
