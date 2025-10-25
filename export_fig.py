from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

ea = event_accumulator.EventAccumulator('./runs/AD-darkroom-seed0')
ea.Reload()  # Load event file

# Get list of all scalar tags
print(ea.Tags()['scalars'])

# Extract one scalar (e.g. train/loss)
events = ea.Scalars('train/loss')

steps = [e.step for e in events]
values = [e.value for e in events]

plt.plot(steps, values)
plt.xlabel("Step")
plt.ylabel("Train Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.savefig('./figs/training_loss.png')
plt.close()

events = ea.Scalars('test/loss_action')


steps = [e.step for e in events]
values = [e.value for e in events]

plt.plot(steps, values)
plt.xlabel("Step")
plt.ylabel("Test Loss")
plt.title("Testing Loss Curve")
plt.grid(True)
plt.savefig('./figs/testing_loss.png')
plt.close()

events = ea.Scalars('train/lr')

steps = [e.step for e in events]
values = [e.value for e in events]

plt.plot(steps, values)
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.grid(True)
plt.savefig('./figs/lr_schedule.png')