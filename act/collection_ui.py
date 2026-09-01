"""Pure terminal prompts for collection review; intentionally ROS-free."""


def prompt_episode_decision(input_fn=input):
    while True:
        decision = input_fn('Episode finished: [s]ave, [d]iscard, [q]uit: ').strip().lower()
        if decision in {'s', 'd', 'q'}:
            return decision
        print('Invalid choice. Enter s, d, or q.')


def prompt_next_decision(next_episode, input_fn=input):
    while True:
        decision = input_fn(
            f'Ready for episode {next_episode}: [n]ext, [q]uit: '
        ).strip().lower()
        if decision in {'n', 'q'}:
            return decision
        print('Invalid choice. Enter n or q.')
