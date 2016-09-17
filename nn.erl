-module(nn).
-compile([export_all]).

-define(ITERATIONS, 10000).
-define(TRAINING_INPUTS, [
		[0,0,0,0],
		[0,0,0,1],
		[0,0,1,0],
		[0,1,0,0],
		[1,0,0,0],
		[0,0,1,1],
		[0,1,1,1],
		[1,1,1,1],
		[1,1,1,0],
		[1,1,0,0],
		[1,1,1,1],
		[1,0,1,0],
		[0,1,0,1],
		[1,0,0,1]
	]).
-define(TRAINING_OUTPUTS, [
		0,
		0,
		0,
		0,
		1,
		0,
		0,
		1,
		1,
		1,
		1,
		1,
		0,
		1
	]).

start() ->
	random:seed(now()),
	random:seed(now()),
	StartingWeights = getRandomStartingSynapticWeights(length(hd(?TRAINING_INPUTS))),
	FinalSynapticWeights = trainWeights(?TRAINING_INPUTS, ?TRAINING_OUTPUTS, StartingWeights, ?ITERATIONS).
	think([0, 1, 1, 0], FinalSynapticWeights).

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
think([TrainingInput1,TrainingInput2, TrainingInput3, TrainingInput4], [SynapticWeight1, SynapticWeight2, SynapticWeight3, SynapticWeight4]) ->
	Sum = (TrainingInput1 * SynapticWeight1) + (TrainingInput2 + SynapticWeight2) + (TrainingInput3 * SynapticWeight3) + (TrainingInput4 + SynapticWeight4),
	IntSum = convertSumToInteger(Sum),
	getSigmoid(IntSum).

%This is just dumb erlang conversion stuff
%basically when you run this neural network with inputs of 4 (eg [1, 1, 0, 0])
%math:exp gets pissed off that the number passed to it is a decimal so I have to
%do this shit to make it an integer
convertSumToInteger(Sum) ->
	%for a sum of 709.828783647
	ListSum = float_to_list(Sum,[{decimals,0}]), %ListSum = "710"
	{IntSum, _} = string:to_integer(ListSum), %This returns {710, []}, hence the pattern matching
	IntSum.

getError(TrainingOutput, Output) ->
	TrainingOutput - Output.

getAdjustedSynapticWeight([SynapticWeight1, SynapticWeight2, SynapticWeight3, SynapticWeight4], [TrainingInput1,TrainingInput2, TrainingInput3, TrainingInput4], Error, Output) ->
	Adjustment1 = TrainingInput1 * (Error * getSigmoidDerivative(Output)),
	Adjustment2 = TrainingInput2 * (Error * getSigmoidDerivative(Output)),
	Adjustment3 = TrainingInput3 * (Error * getSigmoidDerivative(Output)),
	Adjustment4 = TrainingInput4 * (Error * getSigmoidDerivative(Output)),
	[SynapticWeight1 + Adjustment1, SynapticWeight2 + Adjustment2, SynapticWeight3 + Adjustment3, SynapticWeight4 + Adjustment4].
	
%The sigmoid function which describes an S shaped curve
%We pass the weighted sum of the inputs through this function to normalise them between 0-1
getSigmoid(X) ->
	1 / (1 + math:exp(-X)). 

%The derivative of the sigmoid function (gradient of the sigmoid curve)
%indicates how confident we are about the existing weight
getSigmoidDerivative(X) ->
	X * (1 - X).



