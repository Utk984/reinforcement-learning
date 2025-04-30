# Journey of training through multimedia

### 1. Failure

![Fail](fail.gif)


### 2. SO SO SO Many tries

Initially, we saw a lot of erratic movement, so we tried implementing penalties in the action and joint spaces. (Needless to say, it didn't work.)

1. Action Penalty --> arms freeze

![So many tries1](action_penalty.gif)

2. Joint Penalty --> Joints freeze

![So many tries2](joint_penalty.gif)

Eventually, we started realizing that our dense reward function was not working as expected. We had to change the reward function to a sparse reward function.

3. Gripper Open (but balances the needle)
   
![So many tries3](gripper_open_balance.gif)

4. Holding the needle (somewhat)
   
![So many tries4](media/hold_work.gif)

5. Keep trying, and you'll eventually succeed

![So many tries5](media/triple_pickup.gif)


### 3. Eventual Success (this was just our best result)

![Success](media/work_demo.gif)
