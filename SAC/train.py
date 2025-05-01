from env import TextAdventureEnv
from llama import LLaMAWrapper
from ac import Actor, Critic
from replay import ReplayBuffer
from copy import deepcopy
import torch
from torch import nn

env = TextAdventureEnv()
llama = LLaMAWrapper()
actor = Actor(state_dim=4096, action_dim=512)
critic = Critic(state_dim=4096, action_dim=512)
target_critic = deepcopy(critic)

replay_buffer = ReplayBuffer()

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=1e-4)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-4)

# Pre-encode all actions in the action bank
action_texts = env.valid_actions
action_embeddings = torch.stack([llama.encode_text(a).squeeze(0) for a in action_texts])

for episode in range(1000):
    state_text = env.reset()
    done = False

    while not done:
        state_embedding = llama.encode_text(state_text).squeeze(0)
        action_embedding = actor(state_embedding)

        action_text = llama.decode_action(action_embedding, action_embeddings, action_texts)

        next_state_text, reward, done, _ = env.step(action_text)
        next_state_embedding = llama.encode_text(next_state_text).squeeze(0)

        replay_buffer.add((state_embedding.detach(), action_embedding.detach(), reward, next_state_embedding.detach(), done))

        # Training
        if len(replay_buffer.buffer) > 32:
            batch = replay_buffer.sample(32)
            s, a, r, s_next, d = zip(*batch)
            s, a, r, s_next, d = map(torch.stack, (s, a, torch.tensor(r).unsqueeze(1), s_next, torch.tensor(d).unsqueeze(1).float()))

            with torch.no_grad():
                a_next = actor(s_next)
                q_target = r + 0.99 * (1 - d) * target_critic(s_next, a_next)

            q_current = critic(s, a)
            critic_loss = nn.MSELoss()(q_current, q_target)
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            a_pred = actor(s)
            actor_loss = -critic(s, a_pred).mean()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            # Soft update
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

        state_text = next_state_text
