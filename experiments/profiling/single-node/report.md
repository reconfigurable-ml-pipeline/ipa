

# Check the effect of CPU allocation on nodes of different pipelines
## Outcome


## Done experiments

## Series 1
1. audio models
2. check cpu variations with fixing arrival rate to 1 and high memory usage

### Outcome
1. CPU allocation has a linear trend on the model latency
2. Some heavy models are not working at all under some CPU allocation
3. Second run of the experiments has slightly better latency than the first one

## Series 2
1. resnet
2. checking all variables together which didin't work
3. should fixate on some variable and change only some of them
4. series one was done after two

## Series 3
1. resnet
2. check cpu variations
### Outcome
1. CPU allocation has a linear trend on the model latency
2. Some heavy models are not working at all under some CPU allocation
3. Second run of the experiments has slightly better latency than the first one

## Series 4
1. yolo
2. check cpu variations
### Outcome
1. batch size has a linear trend on the model latency
TODO see it's effect on throughtput
TODO find a way to measere queueing latency

## Series 5
1. resnet
2. check batch variations
### Outcome
1. batch size has a linear trend on the model latency
TODO see it's effect on throughtpu
TODO find a way to measere queueing latency

## Series 6
1. resnet
2. check batch variations
3. needs to be repeated as the time of the experiment in previous case was 15s

## Series 7
1. Repeat of the 6 with duration of 60s
2. Should be repeated as there are some Nans

## Series 8
1. Sentiment analysis
2. Should be repeated as there are some Nans

## Series 9
1. sentiment analysis
2. check batch variations

## Series 10
1. audio
2. check batch size variations

## Notes
1. what should I do for models that are not working at all in some experiments?
2. why second run is better?
3. for memory we need to find the "just enough memory" and no optimization after that makes real sense
4. There isn't much change in interesting trend  model 1 in series 8, average latency is almost the same across all batch sizes

## TODO
1. Draw Morphling figures on my experiments
2. Draw RIM figures on my experiments
3. Do effect of changing model_variant/batch/#cpu/hardware/


# Check the effect of batch size on differnt nodes of the pipelines
## Outcome
