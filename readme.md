# Bi-level Actor-Critic for Multi-agent Coordination

This is the code for implementing the MADDPG-based algorithms (Bi-AC, MADDPG) presented in the paper:
[Bi-level Actor-Critic for Multi-agent Coordination](https://arxiv.org/pdf/1909.03510.pdf).

It is base on the Multi-Agent Reinforcement Learning Framework:
[malib](https://github.com/ying-wen/malib/).

It is configured to be run in conjunction with a slightly changed environment, original from the
[highway-env](https://github.com/eleurent/highway-env).

## Installation

- To install, you need to follow the same routine to install [malib](https://github.com/ying-wen/malib/). 

- Main dependencies: Python (3.6), OpenAI gym (0.14.0), tensorflow (2.0.0), numpy (1.17.0), matplotlib, pickle.

## Quick Start

### bilevel_pg

This is the menu for the matrix game setting shown in the paper. To run the experiment in this menu, run:

```shell
cd bilevel_pg/experiments
python run_trainer.py
 ```

### bilevel_pg_highway_1x1

This is the menu for the highway-env setting shown in the paper. To run the experiment in this menu, run:

```shell
cd bilevel_pg_highway_1x1/bilevel_pg
```

Thus, you enter the menu where all the training code are given, you may any of the algorithms given. For example, for running Bi-AC:

```shell
python run_trainer_highway.py
 ```


### bully_q

This is the menu where we test the Bi-Q method without neural netowrk. To run the experiment for Bi-Q, run:

```shell
cd bully_q
python bilevelq_vs_table_q.py
```

## Paper citation

If you used this code for your experiments or found it helpful, consider citing the following paper:

<pre>
@article{zhang2019bi,
  title={Bi-level Actor-Critic for Multi-agent Coordination},
  author={Zhang, Haifeng and Chen, Weizhe and Huang, Zeren and Li, Minne and Yang, Yaodong and Zhang, Weinan and Wang, Jun},
  journal={arXiv preprint arXiv:1909.03510},
  year={2019}
}

</pre>
