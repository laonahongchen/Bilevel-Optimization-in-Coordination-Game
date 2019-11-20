import numpy as np 
import matplotlib.pyplot as plt
seed = 160
lst = np.arange(160, 170)
reward_b_seeds = []
reward_m_seeds = []
reward_i_seeds = []
success_b_seeds = []
success_m_seeds = []
success_i_seeds = []
target_b_seeds = []
target_m_seeds = []
target_i_seeds = []
test_episode = 1000
reward_sum_m_0 = 0
reward_sum_b_0 = 0
reward_sum_i_0 = 0
reward_sum_t_0 = 0
reward_sum_m_1 = 0
reward_sum_b_1 = 0
reward_sum_i_1 = 0
reward_sum_t_1 = 0
success_sum_m = 0
success_sum_b = 0
success_sum_i = 0
target_sum_b = 0
target_sum_m = 0
target_sum_i = 0
target_sum_t = 0
success_sum_b = 0
success_sum_m = 0
success_sum_i = 0
success_sum_t = 0
num_of_seed = 10
for seed in lst:
    reward_b_0 = np.load('./curves/reward0_BILEVEL_1x1_test'+str(seed)+'_s6_t15.npy')
    reward_m_0 = np.load('./curves/reward0_MADDPG_1x1_test'+str(seed)+'_s6_t14.npy')
    reward_i_0 = np.load('./curves/reward0_IQ_1x1_test'+str(seed)+'_s6_t15.npy')
    reward_t_0 = np.load('./curves/reward0_TD3_1x1_test'+str(seed)+'_s6_t15.npy')
    reward_b_1 = np.load('./curves/reward1_BILEVEL_1x1_test'+str(seed)+'_s6_t15.npy')
    reward_m_1 = np.load('./curves/reward1_MADDPG_1x1_test'+str(seed)+'_s6_t14.npy')
    reward_i_1 = np.load('./curves/reward1_IQ_1x1_test'+str(seed)+'_s6_t15.npy')
    reward_t_1 = np.load('./curves/reward1_TD3_1x1_test'+str(seed)+'_s6_t15.npy')
    success_t = np.load('./curves/success_merge_TD3_1x1_test'+str(seed)+'_s6_t15.npy')
    success_b = np.load('./curves/success_merge_BILEVEL_1x1_test'+str(seed)+'_s6_t15.npy')
    success_m = np.load('./curves/success_merge_MADDPG_1x1_test'+str(seed)+'_s6_t14.npy')
    success_i = np.load('./curves/success_merge_IQ_1x1_test'+str(seed)+'_s6_t15.npy')
    target_t = np.load('./curves/target_merge_TD3_1x1_test'+str(seed)+'_s6_t15.npy')
    target_b = np.load('./curves/target_merge_BILEVEL_1x1_test'+str(seed)+'_s6_t15.npy')
    target_m = np.load('./curves/target_merge_MADDPG_1x1_test'+str(seed)+'_s6_t14.npy')
    target_i = np.load('./curves/target_merge_IQ_1x1_test'+str(seed)+'_s6_t15.npy')
    
    reward_sum_b_0 += np.sum(reward_b_0[-test_episode:])
    reward_sum_m_0 += np.sum(reward_m_0[-test_episode:])
    reward_sum_i_0 += np.sum(reward_i_0[-test_episode:])
    reward_sum_t_0 += np.sum(reward_t_0[-test_episode:])
    reward_sum_b_1 += np.sum(reward_b_1[-test_episode:])
    reward_sum_m_1 += np.sum(reward_m_1[-test_episode:])
    reward_sum_i_1 += np.sum(reward_i_1[-test_episode:])
    reward_sum_t_1 += np.sum(reward_t_1[-test_episode:])
    success_sum_b += np.sum(success_b[-test_episode:])
    success_sum_m += np.sum(success_m[-test_episode:])
    success_sum_i += np.sum(success_i[-test_episode:])
    success_sum_t += np.sum(success_t[-test_episode:])
    target_sum_b +=np.sum(target_b[-test_episode:])
    target_sum_m +=np.sum(target_m[-test_episode:])
    target_sum_i +=np.sum(target_i[-test_episode:])
    target_sum_t +=np.sum(target_t[-test_episode:])
    reward_b_seeds.append(reward_b_0)
    reward_m_seeds.append(reward_m_0)
    reward_i_seeds.append(reward_i_0)
    
    success_b_seeds.append(success_b)
    success_m_seeds.append(success_m)
    success_i_seeds.append(success_i)
    target_b_seeds.append(target_b)
    target_m_seeds.append(target_m)
    target_i_seeds.append(target_i)
    
    #print(len(reward_m), len(success_m), len(reward_i), len(success_i))


print("mean reward for b upper agent = ", reward_sum_b_0 / (test_episode * num_of_seed))
print("mean reward for m upper agent = ", reward_sum_m_0 / (test_episode * num_of_seed))
print("mean reward for i upper agent = ", reward_sum_i_0 / (test_episode * num_of_seed))
print("mean reward for t upper agent = ", reward_sum_t_0 / (test_episode * num_of_seed))
print("mean reward for b lower agent = ", reward_sum_b_1 / (test_episode * num_of_seed))
print("mean reward for m lower agent = ", reward_sum_m_1 / (test_episode * num_of_seed))
print("mean reward for i lower agent = ", reward_sum_i_1 / (test_episode * num_of_seed))
print("mean reward for t lower agent = ", reward_sum_t_1 / (test_episode * num_of_seed))
#print("mean success count for b = ", success_sum_b / (test_episode * num_of_seed))
#print("mean success count for m = ", success_sum_m / (test_episode * num_of_seed))
#print("mean success count for i = ", success_sum_i / (test_episode * num_of_seed))
print("mean target rate for b = ", target_sum_b / (test_episode * num_of_seed))
print("mean target rate for m = ", target_sum_m / (test_episode * num_of_seed))
print("mean target rate for i = ", target_sum_i / (test_episode * num_of_seed))
print("mean target rate for t = ", target_sum_t / (test_episode * num_of_seed))
print("mean fail rate for b = ", 1- success_sum_b / (test_episode * num_of_seed))
print("mean fail rate for m = ", 1 - success_sum_m / (test_episode * num_of_seed))
print("mean fail rate for i = ", 1 - success_sum_i / (test_episode * num_of_seed))
print("mean fail rate for t = ", 1 - success_sum_t / (test_episode * num_of_seed))
print("mean untargeted rate for b = ", success_sum_b / (test_episode * num_of_seed) - target_sum_b / (test_episode * num_of_seed))
print("mean untargeted rate for m = ", success_sum_m / (test_episode * num_of_seed) - target_sum_m / (test_episode * num_of_seed))
print("mean untargeted rate for i = ", success_sum_i / (test_episode * num_of_seed) - target_sum_i / (test_episode * num_of_seed))
print("mean untargeted rate for t = ", success_sum_t / (test_episode * num_of_seed) - target_sum_t / (test_episode * num_of_seed))
print(len(reward_m_seeds))

reward_seeds = []
success_seeds = []
target_seeds = []
#reward_seeds.append(reward_b_seeds)
#reward_seeds.append(reward_m_seeds)
#reward_seeds.append(reward_i_seeds)
success_seeds.append(success_b_seeds)
print(len(success_b_seeds))
success_seeds.append(success_m_seeds)
#success_seeds.append(success_i_seeds)
target_seeds.append(target_b_seeds)
target_seeds.append(target_m_seeds)
#target_seeds.append(target_i_seeds)


success_rate_b = []
success_rate_i = []
success_rate_m = []
success_count_b = 0
success_count_m = 0
success_count_i = 0 


#print(len(success_rate_bilevel), len(success_rate_iq), len(success_rate_maddpg))

errors = []
means = []

means_all = []
sums_all = []
errors_all = []

means_b = []
means_m = []
means_i = []
errors_b = []
errors_m = []
errors_i = []



num_of_algorihtm = 2
length = 7999
mean_count = 1000
reward_means = np.zeros((num_of_algorihtm, length - mean_count))
reward_errors = np.zeros((num_of_algorihtm, length - mean_count ))
success_means = np.zeros((num_of_algorihtm, length - mean_count ))
success_errors = np.zeros((num_of_algorihtm, length - mean_count))
target_means = np.zeros((num_of_algorihtm, length - mean_count ))
target_errors = np.zeros((num_of_algorihtm, length - mean_count ))
non_target_means = np.zeros((num_of_algorihtm, length - mean_count))
non_target_errors = np.zeros((num_of_algorihtm, length - mean_count))
fail_means = np.zeros((num_of_algorihtm, length - mean_count))
fail_errors = np.zeros((num_of_algorihtm, length - mean_count))


   

for i in range(num_of_algorihtm):
    for k in range(mean_count, length):
        reward_sums = []
        success_sums = []
        target_sums = []
        non_target_sums = []
        fail_sums = []
        for j in range(num_of_seed):
            #reward_sums.append(np.mean(reward_seeds[i][j][k-mean_count:k]))
            success_sums.append(np.mean(success_seeds[i][j][k-mean_count:k]))
            target_sums.append(np.mean(target_seeds[i][j][k-mean_count:k]))
            non_target_sums.append(np.mean(success_seeds[i][j][k-mean_count:k]) - np.mean(target_seeds[i][j][k-mean_count:k]))
            
            fail_sums.append(1 - np.mean(success_seeds[i][j][k-mean_count:k]))
        #print(np.mean(reward_sums))
        #print(np.mean(reward_sums))
        #reward_means[i][k - mean_count] = np.mean(reward_sums)
        #print(reward_means)
        #print(reward_means[i])
        #reward_errors[i][k - mean_count] = np.std(reward_sums)
        target_means[i][k - mean_count] = np.mean(target_sums)
        target_errors[i][k - mean_count] = np.std(target_sums)
        success_means[i][k - mean_count] = np.mean(success_sums)
        success_errors[i][k - mean_count] = np.std(success_sums)   
        non_target_means[i][k - mean_count] = np.mean(non_target_sums)
        non_target_errors[i][k - mean_count] = np.std(non_target_sums)
        fail_means[i][k - mean_count] = np.mean(fail_sums)
        fail_errors[i][k - mean_count] = np.std(fail_sums)
        #print(len(reward))
        #print(reward_means)
print(len(reward_means[0]))

episodes = np.arange(mean_count, length)


reward_means = np.array(reward_means)
reward_errors = np.array(reward_errors)
success_means = np.array(success_means)
success_errors = np.array(success_errors)
target_means = np.array(target_means)
target_errors = np.array(target_errors)
non_target_means = np.array(non_target_means)
non_target_errors = np.array(non_target_errors)
fail_means = np.array(fail_means)
fail_errors = np.array(fail_errors)

#Reward Curve

#plt.plot(episodes, reward_means[0], label='bilevel')

'''
plt.plot(episodes, reward_means[1], label='maddpg')
plt.plot(episodes, reward_means[2], label='independent q')
plt.fill_between(episodes, reward_means[0]-reward_errors[0], reward_means[0] + reward_errors[0], alpha=0.2)
plt.fill_between(episodes, reward_means[1]-reward_errors[1], reward_means[1] + reward_errors[1], alpha=0.2)
plt.fill_between(episodes, reward_means[2]-reward_errors[2], reward_means[2] + reward_errors[2], alpha=0.2)
'''


#plt.plot(episodes, non_target_means[0], label='Non-target merge')
#plt.fill_between(episodes, fail_means[0]-fail_errors[0], fail_means[0] + fail_errors[0], alpha=0.2)
plt.subplot(1, 2, 1)
plt.ylim(0, 1)
plt.xlim(1000, 4500)
plt.xlabel('Episode',{'size':16})
plt.ylabel('Rate', {'size':16})
plt.tick_params(labelsize=14)
plt.fill_between(episodes, 0, non_target_means[0], color='wheat')
plt.fill_between(episodes, non_target_means[0], non_target_means[0] + fail_means[0], color='lightpink')
plt.fill_between(episodes, non_target_means[0] + fail_means[0], 1, color='lightgreen')

#plt.plot(episodes, reward_means[1], label='maddpg')
#plt.plot(episodes, reward_means[2], label='independent q')
plt.legend(['Follower go first', 'Crash', 'Leader go first'], loc=1)
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.ylim(0, 1)
plt.xlim(1000, 4500)
plt.xlabel('Episode',{'size':16})
plt.ylabel('Rate', {'size':16})
plt.tick_params(labelsize=14)
plt.fill_between(episodes, 0, non_target_means[1], color='wheat')
plt.fill_between(episodes, non_target_means[1], non_target_means[1] + fail_means[1], color='lightpink')
plt.fill_between(episodes, non_target_means[1] + fail_means[1], 1, color='lightgreen')

#plt.plot(episodes, reward_means[1], label='maddpg')
#plt.plot(episodes, reward_means[2], label='independent q')
plt.legend(['Follower go first', 'Crash', 'Leader go first'], loc=1)
plt.tight_layout()

#plt.show()
plt.savefig("1x1_merge.pdf")
plt.show()
