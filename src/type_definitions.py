from typing import List, Tuple, TypeVar

StateType = TypeVar('StateType')
ActionType = TypeVar('ActionType')

TransitionType = Tuple[StateType, ActionType, float]
TrajectoryType = List[TransitionType]

FunctionApproximatorType = TypeVar('FunctionApproximatorType')
