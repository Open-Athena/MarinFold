## Exp124: pause-token training for contact prediction


- Test the pause-token idea from Goyal et al. (ICLR 2024) in contacts-v1.

- Insert <think> tokens during training while masking their prediction loss.

- The full v5p-128 run completed and exported step 35680.

- Evaluate 100 rollouts per protein on the legacy 554-protein contact set.

- Conclusion: no contact-prediction improvement in this tested setting.

- README is the canonical record; committed CSVs provide the reported values.

## Validation losses: compare matching loss definitions


- Ordinary contacts-v1 validation: exp124 = 3.131303071975708 nats.

- Same-stack exp177 CE reference: approximately 3.119; exp124 is about 0.012 worse.

- Historical #117/#75 losses (2.7037 / 2.7566) use an older loss scale.

- Those historical values are not direct absolute-loss baselines for exp124.

- On the same think-masked cache: exp124 = 3.0855870246887207.

- Recomputed #117 on that cache = 3.099645097316767.

- The native masked-loss gain did not transfer to downstream contact accuracy.

## Pause tokens do not improve contact accuracy


- Standard prompt: all R = 0.3365; long R = 0.2856; all AUC = 0.8119.

- One <think>: all R = 0.3359; long R = 0.2822; all AUC = 0.8103.

- Two <think>: all R = 0.3347; long R = 0.2805; all AUC = 0.8093.

- Three <think>: all R = 0.3353; long R = 0.2806; all AUC = 0.8093.

- Long recall@L is lower than the standard prompt in every inserted-token setting.

- This small contacts-v1 experiment did not reproduce the reported pause-token gain.
