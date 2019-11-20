"""
The trainer for multi-agent training.
"""
import pickle
# from malib.trainers.utils import *
from bilevel_pg.bilevelpg.trainer.utils import *
import time

class Bilevel_Trainer:
    """This class implements a multi-agent trainer.
    """
    def __init__(
            self, env, agents, sampler,
            batch_size=128,
            steps=10000,
            exploration_steps=100,
            training_interval=1,
            extra_experiences=['target_actions'],
            save_path=None,
    ):
        self.env = env
        self.agents = agents
        self.sampler = sampler
        self.batch_size = batch_size
        self.steps = steps
        self.exploration_steps = exploration_steps
        self.training_interval = training_interval
        # print(training_interval)
        self.extra_experiences = extra_experiences
        self.losses = []
        self.save_path = save_path

    def setup(self, env, agents, sampler):
        self.env = env
        self.agents = agents
        self.sampler = sampler

    def sample_batches(self):
        assert len(self.agents) > 1
        batches = []
        indices = self.agents[0].replay_buffer.random_indices(self.batch_size)
        for agent in self.agents:
            batch = agent.replay_buffer.batch_by_indices(indices)
            batches.append(batch)
        return batches

    def do_communication(self):
        pass

    def individual_forward(self):
        pass

    def centralized_forward(self):
        pass

    def apply_gradient(self):
        pass

    def run(self):
        print('trainer_start')
        prt_x = []
        prt_y_1 = []
        prt_y_2 = []
        prt_y_3 = []
        action_0 = []
        prt_y_4 = []
        # filewriter = tf.summary.FileWriter('logs')
        for step in range(self.steps):
            if step < self.exploration_steps:
                self.sampler.sample(explore=True)
                continue
            self.sampler.sample()

            batches = self.sample_batches()
            # print(np.array(batches)[0])
            # print('sample finish')
            # print(int(round(time.time() * 1000)))

            for extra_experience in self.extra_experiences:
                if extra_experience == 'annealing':
                    batches = add_annealing(batches, step, annealing_scale=1.)
                elif extra_experience == 'target_actions':
                    batches = add_target_actions(batches, self.agents, self.batch_size)
                elif extra_experience == 'target_actions_no_target':
                    batches = add_target_actions(batches, self.agents, self.batch_size, use_target=False)
                elif extra_experience == 'target_actions_q_pg':
                    batches = add_target_actions_q_pg(batches, self.agents, self.batch_size)
                elif extra_experience == 'target_actions_pg_2':
                    batches = add_target_actions_pg_2(batches, self.agents, self.batch_size)
                elif extra_experience == 'target_actions_pg_2_con':
                    batches = add_target_actions_pg_2_continuous(batches, self.agents, self.batch_size)
                elif extra_experience == 'inner_products':
                    batches = add_inner_product(batches, self.agents, self.batch_size)
                elif extra_experience == 'recent_experiences':
                    batches = add_recent_batches(batches, self.agents, self.batch_size)
                elif extra_experience == 'target_actions_inner':
                    batches = add_target_actions_inner(batches, self.agents, self.batch_size)
            agents_losses = []

            # print('extra finish')
            # print(int(round(time.time() * 1000)))

            if step % self.training_interval == 0:
                for agent, batch in zip(self.agents, batches):
                    # print(batch['actions'].shape)
                    agent_losses = agent.train(batch)
                    agents_losses.append(agent_losses)
                # prt_x.append(step-self.exploration_steps)
                # prt_y_1.append(self.agents[1].get_critic_value(np.array([[1, 0, 0, 1, 0, 0, 1]]))[0][0])
                # prt_y_3.append(self.agents[0].get_critic_value(np.array([[1, 0, 0, 1, 0, 0, 1]]))[0][0])
                # prt_y_2.append(self.agents[1].get_policy_np(np.array([[1, 0, 0, 1]]))[0][2])
                # cnt = 0
                # for i in range(100):
                #     if action_0[step - i] == 2:
                #         cnt = cnt + 1
                # prt_y_4.append(cnt/100)
                # print(self.agents[0].get_critic_value(np.array([[1, 1, 0, 0, 1, 0, 0]])))
                # tf.summary.scalar('q-0-0.0', self.agents[0].get_critic_value(np.array([[1, 1, 0, 0, 1, 0, 0]]))[0][0])
                # print(self.agents[0].get_critic_value(np.array([[1, 0, 0, 1, 1, 0, 0]])))
                # tf.summary.scalar('q-0-2.0', self.agents[0].get_critic_value(np.array([[1, 0, 0, 1, 1, 0, 0]]))[0][0])
                # print(self.agents[0].get_critic_value(np.array([[1, 0, 0, 1, 0, 1, 0]])))
                # tf.summary.scalar('q-0-2.1', self.agents[0].get_critic_value(np.array([[1, 0, 0, 1, 0, 1, 0]]))[0][0])
                # print(self.agents[0].get_critic_value(np.array([[1, 0, 0, 1, 0, 0, 1]])))
                # tf.summary.scalar('q-0-2.2', self.agents[0].get_critic_value(np.array([[1, 0, 0, 1, 0, 0, 1]]))[0][0])
                # merged = tf.summary.merge_all()
                # print(self.agents[1].get_critic_value(np.array([[1, 0, 0, 1, 1, 0, 0]])))
                # print(self.agents[1].get_critic_value(np.array([[1, 0, 0, 1, 0, 1, 0]])))
                # print(self.agents[1].get_critic_value(np.array([[1, 0, 0, 1, 0, 0, 1]])))
                # print(self.agents[0].get_critic_value(np.array([[1, 0, 1, 0, 1]])))
                # print(self.agents[1].get_critic_value(np.array([[1, 1, 0, 1, 0]])))
                # print(self.agents[1].get_critic_value(np.array([[1, 1, 0, 0, 1]])))
                # print(self.agents[1].get_critic_value(np.array([[1, 0, 1, 1, 0]])))
                # print(self.agents[1].get_critic_value(np.array([[1, 0, 1, 0, 1]])))
                # print(self.agents[1].get_critic_value(np.hstack((tf.one_hot([44], self.env.num_state), [[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]]))))
                # print(self.agents[0].get_critic_value(np.hstack((tf.one_hot([44], self.env.num_state), [[0, 1, 0, 0, 1, 0, 0, 0]]))))
                # print(self.agents[1].get_critic_value(
                #     np.hstack((tf.one_hot([44], self.env.num_state), [[0, 1, 0, 0, 1, 0, 0, 0]]))))
                # print(self.agents[0].get_critic_value(
                #     np.hstack((tf.one_hot([44], self.env.num_state), [[0, 1, 0, 0, 0, 1, 0, 0]]))))
                # print(self.agents[1].get_critic_value(
                #     np.hstack((tf.one_hot([44], self.env.num_state), [[0, 1, 0, 0, 0, 1, 0, 0]]))))
                # print(self.agents[0].get_critic_value(
                #     np.hstack((tf.one_hot([44], self.env.num_state), [[0, 1, 0, 0, 0, 0, 1, 0]]))))
                # print(self.agents[1].get_critic_value(
                #     np.hstack((tf.one_hot([44], self.env.num_state), [[0, 1, 0, 0, 0, 0, 1, 0]]))))
                # print(self.agents[0].get_critic_value(
                #     np.hstack((tf.one_hot([44], self.env.num_state), [[0, 1, 0, 0, 0, 0, 0, 1]]))))
                # print(self.agents[1].get_critic_value(
                #     np.hstack((tf.one_hot([44], self.env.num_state), [[0, 1, 0, 0, 0, 0, 0, 1]]))))

                # print(self.agents[1].get_policy_np(np.array([[1, 1, 0]])))
                # print(self.agents[1].get_policy_np(np.array([[1, 0, 0, 1]])))
                # tf.summary.scalar('best-response-policy-2.0', self.agents[1].get_policy_np(np.array([[1, 0, 0, 1]]))[0][0])

            self.losses.append(agent_losses)

            # policy_0 = self.agents[0].get_policy_np(np.array([[1]]))[0]
            # policy_1 = self.agents[1].get_policy_np(np.array([[1]]))[0]



            # print(policy_0)
            # print(policy_1)

            # if step == self.steps - 5:
            #     action_0 = self.agents[0].act(np.array([[1]]), self.agents[1])[0]
            #     action_1 = self.agents[1].act(np.array([[1, 0, 0, 1]]))[0]
            #     f = open("bilevel_grid_result.log", 'a')
            #     f.write(str((action_0, action_1)))
            #     f.close()
            #     break

            # print('train finish')
            # print(int(round(time.time() * 1000)))

        # print(self.agents[0].act(np.array([62]), self.agents[1]))
        # return prt_x, prt_y_1, prt_y_2, prt_y_3, prt_y_4

    def save(self):
        if self.save_path is None:
            self.save_path = '/tmp/agents.pickle'
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.agents, f, pickle.HIGHEST_PROTOCOL)

    def restore(self, restore_path):
        with open(restore_path, 'rb') as f:
            self.agents = pickle.load(f)

    def resume(self):
        pass

    def log_diagnostics(self):
        pass