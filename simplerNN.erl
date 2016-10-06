-module(simplerNN).
-compile([export_all]).

-define(ITERATIONS, 10000).

start() ->
	TrainingInputs = [[0, 0], [1, 1], [0, 1]],
	TrainingOutputs = [0, 1, 0],
	random:seed(now()),
	random:seed(now()),
	StartingWeights = getRandomStartingSynapticWeights(length(hd(TrainingInputs))),
	FinalSynapticWeights = trainWeights(TrainingInputs, TrainingOutputs, StartingWeights, ?ITERATIONS),
	think([1, 0], FinalSynapticWeights).

%gets starting weights the same size as the training input sets
getRandomStartingSynapticWeights(N) ->
	getRandomStartingSynapticWeights([random:uniform() - random:uniform()], N-1).

getRandomStartingSynapticWeights(Weights, 0) ->
	Weights;

getRandomStartingSynapticWeights(Weights, N) ->
	getRandomStartingSynapticWeights(Weights ++ [random:uniform() - random:uniform()], N-1).

trainWeights(TrainingInputs, TrainingOutputs, SynapticWeights, N)->
	trainWeights(TrainingInputs, TrainingOutputs, SynapticWeights, 0, N).

trainWeights(_, _, SynapticWeights, N, N) ->
	SynapticWeights;

trainWeights(TrainingInputs, TrainingOutputs, SynapticWeights, Count, N) ->
	NewSynapticWeights = trainWeightsOnSet(TrainingInputs, TrainingOutputs, SynapticWeights),
	trainWeights(TrainingInputs, TrainingOutputs, NewSynapticWeights, Count+1, N).

trainWeightsOnSet([], [], NewSynapticWeights) ->
	NewSynapticWeights;

%TI = Training Input
%TO = Training Output
trainWeightsOnSet([TI|TIs], [TO|TOs], SynapticWeights) ->
	Output = think(TI, SynapticWeights),
	Error = getError(TO, Output),
	NewSynapticWeight = getAdjustedSynapticWeight(SynapticWeights, TI, Error, Output),
	trainWeightsOnSet(TIs, TOs, NewSynapticWeight).

%This is going to output some outputs for the inputs which will be compared with
%the training set outputs later
think([TrainingInput1,TrainingInput2], [SynapticWeight1, SynapticWeight2]) ->
	Sum = (TrainingInput1 * SynapticWeight1) + (TrainingInput2 * SynapticWeight2),
	getSigmoid(Sum).

getError(TrainingOutput, Output) ->
	TrainingOutput - Output.

getAdjustedSynapticWeight([SynapticWeight1, SynapticWeight2], [TrainingInput1,TrainingInput2], Error, Output) ->
	Adjustment1 = TrainingInput1 * (Error * getSigmoidDerivative(Output)),
	Adjustment2 = TrainingInput2 * (Error * getSigmoidDerivative(Output)),
	[SynapticWeight1 + Adjustment1, SynapticWeight2 + Adjustment2].
	
%The sigmoid function which describes an S shaped curve
%We pass the weighted sum of the inputs through this function to normalise them between 0-1
getSigmoid(X) ->
	1 / (1 + math:exp(-X)). 

%The derivative of the sigmoid function (gradient of the sigmoid curve)
%indicates how confident we are about the existing weight
getSigmoidDerivative(X) ->
	X * (1 - X).



