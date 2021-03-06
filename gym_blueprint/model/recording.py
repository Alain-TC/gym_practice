def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
    """Helper function to store score and print statistics.
    Arguments:
      episode: Current episode
      episode_reward: Reward accumulated over the current episode
      worker_idx: Which thread (worker)
      global_ep_reward: The moving average of the global reward
      result_queue: Queue storing the moving average of the scores
      total_loss: The total loss accumualted over the current episode
      num_steps: The number of steps the episode took to complete
    """
    if global_ep_reward == 0:
        global_ep_reward = episode_reward
    else:
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    print("Episode: {} | Moving Average Reward: {} | Episode Reward: {} | Loss: {} |  Steps: {} | Worker: {}".format(
        episode, str(int(global_ep_reward)), str(int(episode_reward)),
        str(int(total_loss / float(num_steps) * 1000) / 1000),num_steps, worker_idx)
    )
    result_queue.put(global_ep_reward)
    return global_ep_reward
