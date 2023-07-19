# MultiLR
A method for assigning separate learning rate schedulers to different parameters group in a model.

## Usage
Write a lambda function to construct a scheduler for each parameter group. 
```
scheduler = MultiLR(optimizer, 
                [lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5), 
                 lambda opt: torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.25, total_iters=10)])
```
