??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
?
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
?
$sequential/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*5
shared_name&$sequential/layer_normalization/gamma
?
8sequential/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp$sequential/layer_normalization/gamma*
_output_shapes	
:?*
dtype0
?
#sequential/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#sequential/layer_normalization/beta
?
7sequential/layer_normalization/beta/Read/ReadVariableOpReadVariableOp#sequential/layer_normalization/beta*
_output_shapes	
:?*
dtype0
?
sequential/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_namesequential/dense/kernel
?
+sequential/dense/kernel/Read/ReadVariableOpReadVariableOpsequential/dense/kernel*
_output_shapes
:	?*
dtype0
?
sequential/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namesequential/dense/bias
{
)sequential/dense/bias/Read/ReadVariableOpReadVariableOpsequential/dense/bias*
_output_shapes
:*
dtype0
?
sequential/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_namesequential/dense_1/kernel
?
-sequential/dense_1/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_1/kernel*
_output_shapes

:*
dtype0
?
sequential/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namesequential/dense_1/bias

+sequential/dense_1/bias/Read/ReadVariableOpReadVariableOpsequential/dense_1/bias*
_output_shapes
:*
dtype0
?
sequential/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_namesequential/dense_2/kernel
?
-sequential/dense_2/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_2/kernel*
_output_shapes

:*
dtype0
?
sequential/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namesequential/dense_2/bias

+sequential/dense_2/bias/Read/ReadVariableOpReadVariableOpsequential/dense_2/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
T

train_step
metadata
model_variables
_all_assets

signatures
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
	3

4
5
6
7

0
 
fd
VARIABLE_VALUE$sequential/layer_normalization/gamma,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE#sequential/layer_normalization/beta,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEsequential/dense/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEsequential/dense/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEsequential/dense_1/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEsequential/dense_1/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEsequential/dense_2/kernel,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEsequential/dense_2/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE

ref
1


_q_network
?
_sequential_layers
_layer_has_state
regularization_losses
	variables
trainable_variables
	keras_api
#
0
1
2
3
4
 
 
8
0
1
2
	3

4
5
6
7
8
0
1
2
	3

4
5
6
7
?
metrics
layer_metrics
layer_regularization_losses

layers
 non_trainable_variables
regularization_losses
	variables
trainable_variables
q
!axis
	gamma
beta
"regularization_losses
#	variables
$trainable_variables
%	keras_api
R
&regularization_losses
'	variables
(trainable_variables
)	keras_api
h

kernel
	bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
h


kernel
bias
.regularization_losses
/	variables
0trainable_variables
1	keras_api
h

kernel
bias
2regularization_losses
3	variables
4trainable_variables
5	keras_api
 
 
 
#
0
1
2
3
4
 
 
 

0
1

0
1
?
6metrics
7layer_metrics
8layer_regularization_losses

9layers
:non_trainable_variables
"regularization_losses
#	variables
$trainable_variables
 
 
 
?
;metrics
<layer_metrics
=layer_regularization_losses

>layers
?non_trainable_variables
&regularization_losses
'	variables
(trainable_variables
 

0
	1

0
	1
?
@metrics
Alayer_metrics
Blayer_regularization_losses

Clayers
Dnon_trainable_variables
*regularization_losses
+	variables
,trainable_variables
 


0
1


0
1
?
Emetrics
Flayer_metrics
Glayer_regularization_losses

Hlayers
Inon_trainable_variables
.regularization_losses
/	variables
0trainable_variables
 

0
1

0
1
?
Jmetrics
Klayer_metrics
Llayer_regularization_losses

Mlayers
Nnon_trainable_variables
2regularization_losses
3	variables
4trainable_variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
l
action_0/discountPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
action_0/observationPlaceholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
j
action_0/rewardPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
m
action_0/step_typePlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_type$sequential/layer_normalization/gamma#sequential/layer_normalization/betasequential/dense/kernelsequential/dense/biassequential/dense_1/kernelsequential/dense_1/biassequential/dense_2/kernelsequential/dense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_41279104
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_41279116
?
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_41279138
?
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_41279131
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp8sequential/layer_normalization/gamma/Read/ReadVariableOp7sequential/layer_normalization/beta/Read/ReadVariableOp+sequential/dense/kernel/Read/ReadVariableOp)sequential/dense/bias/Read/ReadVariableOp-sequential/dense_1/kernel/Read/ReadVariableOp+sequential/dense_1/bias/Read/ReadVariableOp-sequential/dense_2/kernel/Read/ReadVariableOp+sequential/dense_2/bias/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_save_41279426
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable$sequential/layer_normalization/gamma#sequential/layer_normalization/betasequential/dense/kernelsequential/dense/biassequential/dense_1/kernelsequential/dense_1/biassequential/dense_2/kernelsequential/dense_2/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__traced_restore_41279463??
?
l
,__inference_function_with_signature_41279123
unknown:	 
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference_<lambda>_412788182
StatefulPartitionedCallj
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
??
?
*__inference_polymorphic_action_fn_41279223
	step_type

reward
discount
observationM
>sequential_layer_normalization_reshape_readvariableop_resource:	?O
@sequential_layer_normalization_reshape_1_readvariableop_resource:	?B
/sequential_dense_matmul_readvariableop_resource:	?>
0sequential_dense_biasadd_readvariableop_resource:C
1sequential_dense_1_matmul_readvariableop_resource:@
2sequential_dense_1_biasadd_readvariableop_resource:C
1sequential_dense_2_matmul_readvariableop_resource:@
2sequential_dense_2_biasadd_readvariableop_resource:
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?)sequential/dense_2/BiasAdd/ReadVariableOp?(sequential/dense_2/MatMul/ReadVariableOp?5sequential/layer_normalization/Reshape/ReadVariableOp?7sequential/layer_normalization/Reshape_1/ReadVariableOp?
#sequential/layer_normalization/CastCastobservation*

DstT0*

SrcT0*,
_output_shapes
:??????????2%
#sequential/layer_normalization/Cast?
=sequential/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2?
=sequential/layer_normalization/moments/mean/reduction_indices?
+sequential/layer_normalization/moments/meanMean'sequential/layer_normalization/Cast:y:0Fsequential/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2-
+sequential/layer_normalization/moments/mean?
3sequential/layer_normalization/moments/StopGradientStopGradient4sequential/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:?????????25
3sequential/layer_normalization/moments/StopGradient?
8sequential/layer_normalization/moments/SquaredDifferenceSquaredDifference'sequential/layer_normalization/Cast:y:0<sequential/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????2:
8sequential/layer_normalization/moments/SquaredDifference?
Asequential/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2C
Asequential/layer_normalization/moments/variance/reduction_indices?
/sequential/layer_normalization/moments/varianceMean<sequential/layer_normalization/moments/SquaredDifference:z:0Jsequential/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(21
/sequential/layer_normalization/moments/variance?
5sequential/layer_normalization/Reshape/ReadVariableOpReadVariableOp>sequential_layer_normalization_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype027
5sequential/layer_normalization/Reshape/ReadVariableOp?
,sequential/layer_normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?      2.
,sequential/layer_normalization/Reshape/shape?
&sequential/layer_normalization/ReshapeReshape=sequential/layer_normalization/Reshape/ReadVariableOp:value:05sequential/layer_normalization/Reshape/shape:output:0*
T0*#
_output_shapes
:?2(
&sequential/layer_normalization/Reshape?
7sequential/layer_normalization/Reshape_1/ReadVariableOpReadVariableOp@sequential_layer_normalization_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype029
7sequential/layer_normalization/Reshape_1/ReadVariableOp?
.sequential/layer_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?      20
.sequential/layer_normalization/Reshape_1/shape?
(sequential/layer_normalization/Reshape_1Reshape?sequential/layer_normalization/Reshape_1/ReadVariableOp:value:07sequential/layer_normalization/Reshape_1/shape:output:0*
T0*#
_output_shapes
:?2*
(sequential/layer_normalization/Reshape_1?
.sequential/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/layer_normalization/batchnorm/add/y?
,sequential/layer_normalization/batchnorm/addAddV28sequential/layer_normalization/moments/variance:output:07sequential/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????2.
,sequential/layer_normalization/batchnorm/add?
.sequential/layer_normalization/batchnorm/RsqrtRsqrt0sequential/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????20
.sequential/layer_normalization/batchnorm/Rsqrt?
,sequential/layer_normalization/batchnorm/mulMul2sequential/layer_normalization/batchnorm/Rsqrt:y:0/sequential/layer_normalization/Reshape:output:0*
T0*,
_output_shapes
:??????????2.
,sequential/layer_normalization/batchnorm/mul?
.sequential/layer_normalization/batchnorm/mul_1Mul'sequential/layer_normalization/Cast:y:00sequential/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????20
.sequential/layer_normalization/batchnorm/mul_1?
.sequential/layer_normalization/batchnorm/mul_2Mul4sequential/layer_normalization/moments/mean:output:00sequential/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????20
.sequential/layer_normalization/batchnorm/mul_2?
,sequential/layer_normalization/batchnorm/subSub1sequential/layer_normalization/Reshape_1:output:02sequential/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????2.
,sequential/layer_normalization/batchnorm/sub?
.sequential/layer_normalization/batchnorm/add_1AddV22sequential/layer_normalization/batchnorm/mul_1:z:00sequential/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????20
.sequential/layer_normalization/batchnorm/add_1?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape2sequential/layer_normalization/batchnorm/add_1:z:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/BiasAdd?
sequential/dense/TanhTanh!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense/Tanh?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMulsequential/dense/Tanh:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/TanhTanh#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/Tanh?
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOp?
sequential/dense_2/MatMulMatMulsequential/dense_1/Tanh:y:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_2/MatMul?
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOp?
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_2/BiasAdd?
sequential/dense_2/SoftmaxSoftmax#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense_2/Softmax?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax$sequential/dense_2/Softmax:softmax:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????2
Categorical_1/mode/ArgMax?
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol?
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape?
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape~
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/sample/Const?
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Deterministic_1/sample/strided_slice/stack?
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_1?
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2?
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2&
$Deterministic_1/sample/strided_slice?
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s0?
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2+
)Deterministic_1/sample/BroadcastArgs/s0_1?
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs?
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0?
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 2(
&Deterministic_1/sample/concat/values_2?
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis?
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat?
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:?????????2$
"Deterministic_1/sample/BroadcastTo?
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1?
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice_1/stack?
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.Deterministic_1/sample/strided_slice_1/stack_1?
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Deterministic_1/sample/strided_slice_1/stack_2?
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2(
&Deterministic_1/sample/strided_slice_1?
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis?
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1?
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:?????????2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:?????????2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:?????????2
clip_by_valueh
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:?????????2

Identity?
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp6^sequential/layer_normalization/Reshape/ReadVariableOp8^sequential/layer_normalization/Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:?????????:?????????:?????????:??????????: : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2n
5sequential/layer_normalization/Reshape/ReadVariableOp5sequential/layer_normalization/Reshape/ReadVariableOp2r
7sequential/layer_normalization/Reshape_1/ReadVariableOp7sequential/layer_normalization/Reshape_1/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	step_type:KG
#
_output_shapes
:?????????
 
_user_specified_namereward:MI
#
_output_shapes
:?????????
"
_user_specified_name
discount:YU
,
_output_shapes
:??????????
%
_user_specified_nameobservation
?!
?
!__inference__traced_save_41279426
file_prefix'
#savev2_variable_read_readvariableop	C
?savev2_sequential_layer_normalization_gamma_read_readvariableopB
>savev2_sequential_layer_normalization_beta_read_readvariableop6
2savev2_sequential_dense_kernel_read_readvariableop4
0savev2_sequential_dense_bias_read_readvariableop8
4savev2_sequential_dense_1_kernel_read_readvariableop6
2savev2_sequential_dense_1_bias_read_readvariableop8
4savev2_sequential_dense_2_kernel_read_readvariableop6
2savev2_sequential_dense_2_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*?
value?B?
B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop?savev2_sequential_layer_normalization_gamma_read_readvariableop>savev2_sequential_layer_normalization_beta_read_readvariableop2savev2_sequential_dense_kernel_read_readvariableop0savev2_sequential_dense_bias_read_readvariableop4savev2_sequential_dense_1_kernel_read_readvariableop2savev2_sequential_dense_1_bias_read_readvariableop4savev2_sequential_dense_2_kernel_read_readvariableop2savev2_sequential_dense_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*X
_input_shapesG
E: : :?:?:	?:::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 	

_output_shapes
::


_output_shapes
: 
?
8
&__inference_get_initial_state_41279110

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?
8
&__inference_get_initial_state_41279371

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
^

__inference_<lambda>_41278821*(
_construction_contextkEagerRuntime*
_input_shapes 
?
d
__inference_<lambda>_41278818!
readvariableop_resource:	 
identity	??ReadVariableOpp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOp`
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: 2

Identity_
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp
?
?
&__inference_signature_wrapper_41279104
discount
observation

reward
	step_type
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *5
f0R.
,__inference_function_with_signature_412790782
StatefulPartitionedCallw
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:?????????:??????????:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
0/discount:[W
,
_output_shapes
:??????????
'
_user_specified_name0/observation:MI
#
_output_shapes
:?????????
"
_user_specified_name
0/reward:PL
#
_output_shapes
:?????????
%
_user_specified_name0/step_type
?
.
,__inference_function_with_signature_41279134?
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference_<lambda>_412788212
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes 
?
(
&__inference_signature_wrapper_41279138?
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *5
f0R.
,__inference_function_with_signature_412791342
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes 
?^
?
0__inference_polymorphic_distribution_fn_41279368
	step_type

reward
discount
observationM
>sequential_layer_normalization_reshape_readvariableop_resource:	?O
@sequential_layer_normalization_reshape_1_readvariableop_resource:	?B
/sequential_dense_matmul_readvariableop_resource:	?>
0sequential_dense_biasadd_readvariableop_resource:C
1sequential_dense_1_matmul_readvariableop_resource:@
2sequential_dense_1_biasadd_readvariableop_resource:C
1sequential_dense_2_matmul_readvariableop_resource:@
2sequential_dense_2_biasadd_readvariableop_resource:
identity

identity_1

identity_2??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?)sequential/dense_2/BiasAdd/ReadVariableOp?(sequential/dense_2/MatMul/ReadVariableOp?5sequential/layer_normalization/Reshape/ReadVariableOp?7sequential/layer_normalization/Reshape_1/ReadVariableOp?
#sequential/layer_normalization/CastCastobservation*

DstT0*

SrcT0*,
_output_shapes
:??????????2%
#sequential/layer_normalization/Cast?
=sequential/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2?
=sequential/layer_normalization/moments/mean/reduction_indices?
+sequential/layer_normalization/moments/meanMean'sequential/layer_normalization/Cast:y:0Fsequential/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2-
+sequential/layer_normalization/moments/mean?
3sequential/layer_normalization/moments/StopGradientStopGradient4sequential/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:?????????25
3sequential/layer_normalization/moments/StopGradient?
8sequential/layer_normalization/moments/SquaredDifferenceSquaredDifference'sequential/layer_normalization/Cast:y:0<sequential/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????2:
8sequential/layer_normalization/moments/SquaredDifference?
Asequential/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2C
Asequential/layer_normalization/moments/variance/reduction_indices?
/sequential/layer_normalization/moments/varianceMean<sequential/layer_normalization/moments/SquaredDifference:z:0Jsequential/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(21
/sequential/layer_normalization/moments/variance?
5sequential/layer_normalization/Reshape/ReadVariableOpReadVariableOp>sequential_layer_normalization_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype027
5sequential/layer_normalization/Reshape/ReadVariableOp?
,sequential/layer_normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?      2.
,sequential/layer_normalization/Reshape/shape?
&sequential/layer_normalization/ReshapeReshape=sequential/layer_normalization/Reshape/ReadVariableOp:value:05sequential/layer_normalization/Reshape/shape:output:0*
T0*#
_output_shapes
:?2(
&sequential/layer_normalization/Reshape?
7sequential/layer_normalization/Reshape_1/ReadVariableOpReadVariableOp@sequential_layer_normalization_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype029
7sequential/layer_normalization/Reshape_1/ReadVariableOp?
.sequential/layer_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?      20
.sequential/layer_normalization/Reshape_1/shape?
(sequential/layer_normalization/Reshape_1Reshape?sequential/layer_normalization/Reshape_1/ReadVariableOp:value:07sequential/layer_normalization/Reshape_1/shape:output:0*
T0*#
_output_shapes
:?2*
(sequential/layer_normalization/Reshape_1?
.sequential/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/layer_normalization/batchnorm/add/y?
,sequential/layer_normalization/batchnorm/addAddV28sequential/layer_normalization/moments/variance:output:07sequential/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????2.
,sequential/layer_normalization/batchnorm/add?
.sequential/layer_normalization/batchnorm/RsqrtRsqrt0sequential/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????20
.sequential/layer_normalization/batchnorm/Rsqrt?
,sequential/layer_normalization/batchnorm/mulMul2sequential/layer_normalization/batchnorm/Rsqrt:y:0/sequential/layer_normalization/Reshape:output:0*
T0*,
_output_shapes
:??????????2.
,sequential/layer_normalization/batchnorm/mul?
.sequential/layer_normalization/batchnorm/mul_1Mul'sequential/layer_normalization/Cast:y:00sequential/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????20
.sequential/layer_normalization/batchnorm/mul_1?
.sequential/layer_normalization/batchnorm/mul_2Mul4sequential/layer_normalization/moments/mean:output:00sequential/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????20
.sequential/layer_normalization/batchnorm/mul_2?
,sequential/layer_normalization/batchnorm/subSub1sequential/layer_normalization/Reshape_1:output:02sequential/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????2.
,sequential/layer_normalization/batchnorm/sub?
.sequential/layer_normalization/batchnorm/add_1AddV22sequential/layer_normalization/batchnorm/mul_1:z:00sequential/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????20
.sequential/layer_normalization/batchnorm/add_1?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape2sequential/layer_normalization/batchnorm/add_1:z:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/BiasAdd?
sequential/dense/TanhTanh!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense/Tanh?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMulsequential/dense/Tanh:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/TanhTanh#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/Tanh?
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOp?
sequential/dense_2/MatMulMatMulsequential/dense_1/Tanh:y:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_2/MatMul?
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOp?
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_2/BiasAdd?
sequential/dense_2/SoftmaxSoftmax#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense_2/Softmax?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax$sequential/dense_2/Softmax:softmax:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????2
Categorical_1/mode/ArgMax?
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtole
IdentityIdentityDeterministic/atol:output:0^NoOp*
T0*
_output_shapes
: 2

Identityv

Identity_1IdentityCategorical_1/mode/Cast:y:0^NoOp*
T0*#
_output_shapes
:?????????2

Identity_1i

Identity_2IdentityDeterministic/rtol:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_2?
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp6^sequential/layer_normalization/Reshape/ReadVariableOp8^sequential/layer_normalization/Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:?????????:?????????:?????????:??????????: : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2n
5sequential/layer_normalization/Reshape/ReadVariableOp5sequential/layer_normalization/Reshape/ReadVariableOp2r
7sequential/layer_normalization/Reshape_1/ReadVariableOp7sequential/layer_normalization/Reshape_1/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	step_type:KG
#
_output_shapes
:?????????
 
_user_specified_namereward:MI
#
_output_shapes
:?????????
"
_user_specified_name
discount:YU
,
_output_shapes
:??????????
%
_user_specified_nameobservation
??
?
*__inference_polymorphic_action_fn_41279059
	time_step
time_step_1
time_step_2
time_step_3M
>sequential_layer_normalization_reshape_readvariableop_resource:	?O
@sequential_layer_normalization_reshape_1_readvariableop_resource:	?B
/sequential_dense_matmul_readvariableop_resource:	?>
0sequential_dense_biasadd_readvariableop_resource:C
1sequential_dense_1_matmul_readvariableop_resource:@
2sequential_dense_1_biasadd_readvariableop_resource:C
1sequential_dense_2_matmul_readvariableop_resource:@
2sequential_dense_2_biasadd_readvariableop_resource:
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?)sequential/dense_2/BiasAdd/ReadVariableOp?(sequential/dense_2/MatMul/ReadVariableOp?5sequential/layer_normalization/Reshape/ReadVariableOp?7sequential/layer_normalization/Reshape_1/ReadVariableOp?
#sequential/layer_normalization/CastCasttime_step_3*

DstT0*

SrcT0*,
_output_shapes
:??????????2%
#sequential/layer_normalization/Cast?
=sequential/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2?
=sequential/layer_normalization/moments/mean/reduction_indices?
+sequential/layer_normalization/moments/meanMean'sequential/layer_normalization/Cast:y:0Fsequential/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2-
+sequential/layer_normalization/moments/mean?
3sequential/layer_normalization/moments/StopGradientStopGradient4sequential/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:?????????25
3sequential/layer_normalization/moments/StopGradient?
8sequential/layer_normalization/moments/SquaredDifferenceSquaredDifference'sequential/layer_normalization/Cast:y:0<sequential/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????2:
8sequential/layer_normalization/moments/SquaredDifference?
Asequential/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2C
Asequential/layer_normalization/moments/variance/reduction_indices?
/sequential/layer_normalization/moments/varianceMean<sequential/layer_normalization/moments/SquaredDifference:z:0Jsequential/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(21
/sequential/layer_normalization/moments/variance?
5sequential/layer_normalization/Reshape/ReadVariableOpReadVariableOp>sequential_layer_normalization_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype027
5sequential/layer_normalization/Reshape/ReadVariableOp?
,sequential/layer_normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?      2.
,sequential/layer_normalization/Reshape/shape?
&sequential/layer_normalization/ReshapeReshape=sequential/layer_normalization/Reshape/ReadVariableOp:value:05sequential/layer_normalization/Reshape/shape:output:0*
T0*#
_output_shapes
:?2(
&sequential/layer_normalization/Reshape?
7sequential/layer_normalization/Reshape_1/ReadVariableOpReadVariableOp@sequential_layer_normalization_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype029
7sequential/layer_normalization/Reshape_1/ReadVariableOp?
.sequential/layer_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?      20
.sequential/layer_normalization/Reshape_1/shape?
(sequential/layer_normalization/Reshape_1Reshape?sequential/layer_normalization/Reshape_1/ReadVariableOp:value:07sequential/layer_normalization/Reshape_1/shape:output:0*
T0*#
_output_shapes
:?2*
(sequential/layer_normalization/Reshape_1?
.sequential/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/layer_normalization/batchnorm/add/y?
,sequential/layer_normalization/batchnorm/addAddV28sequential/layer_normalization/moments/variance:output:07sequential/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????2.
,sequential/layer_normalization/batchnorm/add?
.sequential/layer_normalization/batchnorm/RsqrtRsqrt0sequential/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????20
.sequential/layer_normalization/batchnorm/Rsqrt?
,sequential/layer_normalization/batchnorm/mulMul2sequential/layer_normalization/batchnorm/Rsqrt:y:0/sequential/layer_normalization/Reshape:output:0*
T0*,
_output_shapes
:??????????2.
,sequential/layer_normalization/batchnorm/mul?
.sequential/layer_normalization/batchnorm/mul_1Mul'sequential/layer_normalization/Cast:y:00sequential/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????20
.sequential/layer_normalization/batchnorm/mul_1?
.sequential/layer_normalization/batchnorm/mul_2Mul4sequential/layer_normalization/moments/mean:output:00sequential/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????20
.sequential/layer_normalization/batchnorm/mul_2?
,sequential/layer_normalization/batchnorm/subSub1sequential/layer_normalization/Reshape_1:output:02sequential/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????2.
,sequential/layer_normalization/batchnorm/sub?
.sequential/layer_normalization/batchnorm/add_1AddV22sequential/layer_normalization/batchnorm/mul_1:z:00sequential/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????20
.sequential/layer_normalization/batchnorm/add_1?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape2sequential/layer_normalization/batchnorm/add_1:z:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/BiasAdd?
sequential/dense/TanhTanh!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense/Tanh?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMulsequential/dense/Tanh:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/TanhTanh#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/Tanh?
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOp?
sequential/dense_2/MatMulMatMulsequential/dense_1/Tanh:y:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_2/MatMul?
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOp?
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_2/BiasAdd?
sequential/dense_2/SoftmaxSoftmax#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense_2/Softmax?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax$sequential/dense_2/Softmax:softmax:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????2
Categorical_1/mode/ArgMax?
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol?
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape?
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape~
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/sample/Const?
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Deterministic_1/sample/strided_slice/stack?
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_1?
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2?
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2&
$Deterministic_1/sample/strided_slice?
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s0?
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2+
)Deterministic_1/sample/BroadcastArgs/s0_1?
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs?
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0?
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 2(
&Deterministic_1/sample/concat/values_2?
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis?
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat?
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:?????????2$
"Deterministic_1/sample/BroadcastTo?
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1?
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice_1/stack?
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.Deterministic_1/sample/strided_slice_1/stack_1?
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Deterministic_1/sample/strided_slice_1/stack_2?
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2(
&Deterministic_1/sample/strided_slice_1?
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis?
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1?
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:?????????2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:?????????2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:?????????2
clip_by_valueh
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:?????????2

Identity?
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp6^sequential/layer_normalization/Reshape/ReadVariableOp8^sequential/layer_normalization/Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:?????????:?????????:?????????:??????????: : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2n
5sequential/layer_normalization/Reshape/ReadVariableOp5sequential/layer_normalization/Reshape/ReadVariableOp2r
7sequential/layer_normalization/Reshape_1/ReadVariableOp7sequential/layer_normalization/Reshape_1/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	time_step:NJ
#
_output_shapes
:?????????
#
_user_specified_name	time_step:NJ
#
_output_shapes
:?????????
#
_user_specified_name	time_step:WS
,
_output_shapes
:??????????
#
_user_specified_name	time_step
?+
?
$__inference__traced_restore_41279463
file_prefix#
assignvariableop_variable:	 F
7assignvariableop_1_sequential_layer_normalization_gamma:	?E
6assignvariableop_2_sequential_layer_normalization_beta:	?=
*assignvariableop_3_sequential_dense_kernel:	?6
(assignvariableop_4_sequential_dense_bias:>
,assignvariableop_5_sequential_dense_1_kernel:8
*assignvariableop_6_sequential_dense_1_bias:>
,assignvariableop_7_sequential_dense_2_kernel:8
*assignvariableop_8_sequential_dense_2_bias:
identity_10??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*?
value?B?
B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*<
_output_shapes*
(::::::::::*
dtypes
2
	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp7assignvariableop_1_sequential_layer_normalization_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp6assignvariableop_2_sequential_layer_normalization_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp*assignvariableop_3_sequential_dense_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp(assignvariableop_4_sequential_dense_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp,assignvariableop_5_sequential_dense_1_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp*assignvariableop_6_sequential_dense_1_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp,assignvariableop_7_sequential_dense_2_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp*assignvariableop_8_sequential_dense_2_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_9Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_9e
Identity_10IdentityIdentity_9:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_10?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_10Identity_10:output:0*'
_input_shapes
: : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_8:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
>
,__inference_function_with_signature_41279111

batch_size?
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_get_initial_state_412791102
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
??
?
*__inference_polymorphic_action_fn_41279308
time_step_step_type
time_step_reward
time_step_discount
time_step_observationM
>sequential_layer_normalization_reshape_readvariableop_resource:	?O
@sequential_layer_normalization_reshape_1_readvariableop_resource:	?B
/sequential_dense_matmul_readvariableop_resource:	?>
0sequential_dense_biasadd_readvariableop_resource:C
1sequential_dense_1_matmul_readvariableop_resource:@
2sequential_dense_1_biasadd_readvariableop_resource:C
1sequential_dense_2_matmul_readvariableop_resource:@
2sequential_dense_2_biasadd_readvariableop_resource:
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?)sequential/dense_2/BiasAdd/ReadVariableOp?(sequential/dense_2/MatMul/ReadVariableOp?5sequential/layer_normalization/Reshape/ReadVariableOp?7sequential/layer_normalization/Reshape_1/ReadVariableOp?
#sequential/layer_normalization/CastCasttime_step_observation*

DstT0*

SrcT0*,
_output_shapes
:??????????2%
#sequential/layer_normalization/Cast?
=sequential/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2?
=sequential/layer_normalization/moments/mean/reduction_indices?
+sequential/layer_normalization/moments/meanMean'sequential/layer_normalization/Cast:y:0Fsequential/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2-
+sequential/layer_normalization/moments/mean?
3sequential/layer_normalization/moments/StopGradientStopGradient4sequential/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:?????????25
3sequential/layer_normalization/moments/StopGradient?
8sequential/layer_normalization/moments/SquaredDifferenceSquaredDifference'sequential/layer_normalization/Cast:y:0<sequential/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????2:
8sequential/layer_normalization/moments/SquaredDifference?
Asequential/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2C
Asequential/layer_normalization/moments/variance/reduction_indices?
/sequential/layer_normalization/moments/varianceMean<sequential/layer_normalization/moments/SquaredDifference:z:0Jsequential/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(21
/sequential/layer_normalization/moments/variance?
5sequential/layer_normalization/Reshape/ReadVariableOpReadVariableOp>sequential_layer_normalization_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype027
5sequential/layer_normalization/Reshape/ReadVariableOp?
,sequential/layer_normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?      2.
,sequential/layer_normalization/Reshape/shape?
&sequential/layer_normalization/ReshapeReshape=sequential/layer_normalization/Reshape/ReadVariableOp:value:05sequential/layer_normalization/Reshape/shape:output:0*
T0*#
_output_shapes
:?2(
&sequential/layer_normalization/Reshape?
7sequential/layer_normalization/Reshape_1/ReadVariableOpReadVariableOp@sequential_layer_normalization_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype029
7sequential/layer_normalization/Reshape_1/ReadVariableOp?
.sequential/layer_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?      20
.sequential/layer_normalization/Reshape_1/shape?
(sequential/layer_normalization/Reshape_1Reshape?sequential/layer_normalization/Reshape_1/ReadVariableOp:value:07sequential/layer_normalization/Reshape_1/shape:output:0*
T0*#
_output_shapes
:?2*
(sequential/layer_normalization/Reshape_1?
.sequential/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/layer_normalization/batchnorm/add/y?
,sequential/layer_normalization/batchnorm/addAddV28sequential/layer_normalization/moments/variance:output:07sequential/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????2.
,sequential/layer_normalization/batchnorm/add?
.sequential/layer_normalization/batchnorm/RsqrtRsqrt0sequential/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????20
.sequential/layer_normalization/batchnorm/Rsqrt?
,sequential/layer_normalization/batchnorm/mulMul2sequential/layer_normalization/batchnorm/Rsqrt:y:0/sequential/layer_normalization/Reshape:output:0*
T0*,
_output_shapes
:??????????2.
,sequential/layer_normalization/batchnorm/mul?
.sequential/layer_normalization/batchnorm/mul_1Mul'sequential/layer_normalization/Cast:y:00sequential/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????20
.sequential/layer_normalization/batchnorm/mul_1?
.sequential/layer_normalization/batchnorm/mul_2Mul4sequential/layer_normalization/moments/mean:output:00sequential/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????20
.sequential/layer_normalization/batchnorm/mul_2?
,sequential/layer_normalization/batchnorm/subSub1sequential/layer_normalization/Reshape_1:output:02sequential/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????2.
,sequential/layer_normalization/batchnorm/sub?
.sequential/layer_normalization/batchnorm/add_1AddV22sequential/layer_normalization/batchnorm/mul_1:z:00sequential/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????20
.sequential/layer_normalization/batchnorm/add_1?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape2sequential/layer_normalization/batchnorm/add_1:z:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/BiasAdd?
sequential/dense/TanhTanh!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense/Tanh?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMulsequential/dense/Tanh:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/TanhTanh#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/Tanh?
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOp?
sequential/dense_2/MatMulMatMulsequential/dense_1/Tanh:y:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_2/MatMul?
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOp?
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_2/BiasAdd?
sequential/dense_2/SoftmaxSoftmax#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense_2/Softmax?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax$sequential/dense_2/Softmax:softmax:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????2
Categorical_1/mode/ArgMax?
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol?
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape?
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape~
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/sample/Const?
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Deterministic_1/sample/strided_slice/stack?
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_1?
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2?
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2&
$Deterministic_1/sample/strided_slice?
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s0?
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2+
)Deterministic_1/sample/BroadcastArgs/s0_1?
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs?
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0?
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 2(
&Deterministic_1/sample/concat/values_2?
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis?
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat?
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:?????????2$
"Deterministic_1/sample/BroadcastTo?
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1?
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice_1/stack?
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.Deterministic_1/sample/strided_slice_1/stack_1?
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Deterministic_1/sample/strided_slice_1/stack_2?
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2(
&Deterministic_1/sample/strided_slice_1?
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis?
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1?
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:?????????2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:?????????2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:?????????2
clip_by_valueh
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:?????????2

Identity?
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp6^sequential/layer_normalization/Reshape/ReadVariableOp8^sequential/layer_normalization/Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:?????????:?????????:?????????:??????????: : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2n
5sequential/layer_normalization/Reshape/ReadVariableOp5sequential/layer_normalization/Reshape/ReadVariableOp2r
7sequential/layer_normalization/Reshape_1/ReadVariableOp7sequential/layer_normalization/Reshape_1/ReadVariableOp:X T
#
_output_shapes
:?????????
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:?????????
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:?????????
,
_user_specified_nametime_step/discount:c_
,
_output_shapes
:??????????
/
_user_specified_nametime_step/observation
?
?
,__inference_function_with_signature_41279078
	step_type

reward
discount
observation
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *3
f.R,
*__inference_polymorphic_action_fn_412790592
StatefulPartitionedCallw
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:?????????:?????????:?????????:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:?????????
%
_user_specified_name0/step_type:MI
#
_output_shapes
:?????????
"
_user_specified_name
0/reward:OK
#
_output_shapes
:?????????
$
_user_specified_name
0/discount:[W
,
_output_shapes
:??????????
'
_user_specified_name0/observation
?
f
&__inference_signature_wrapper_41279131
unknown:	 
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *5
f0R.
,__inference_function_with_signature_412791232
StatefulPartitionedCallj
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
?
8
&__inference_signature_wrapper_41279116

batch_size?
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *5
f0R.
,__inference_function_with_signature_412791112
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
action?
4

0/discount&
action_0/discount:0?????????
C
0/observation2
action_0/observation:0??????????
0
0/reward$
action_0/reward:0?????????
6
0/step_type'
action_0/step_type:0?????????6
action,
StatefulPartitionedCall:0?????????tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:?_
?

train_step
metadata
model_variables
_all_assets

signatures

Oaction
Pdistribution
Qget_initial_state
Rget_metadata
Sget_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
Y
0
1
2
	3

4
5
6
7"
trackable_tuple_wrapper
'
0"
trackable_list_wrapper
`

Taction
Uget_initial_state
Vget_train_step
Wget_metadata"
signature_map
3:1?2$sequential/layer_normalization/gamma
2:0?2#sequential/layer_normalization/beta
*:(	?2sequential/dense/kernel
#:!2sequential/dense/bias
+:)2sequential/dense_1/kernel
%:#2sequential/dense_1/bias
+:)2sequential/dense_2/kernel
%:#2sequential/dense_2/bias
1
ref
1"
trackable_tuple_wrapper
.

_q_network"
_generic_user_object
?
_sequential_layers
_layer_has_state
regularization_losses
	variables
trainable_variables
	keras_api
*X&call_and_return_all_conditional_losses
Y__call__"
_tf_keras_layer
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
	3

4
5
6
7"
trackable_list_wrapper
X
0
1
2
	3

4
5
6
7"
trackable_list_wrapper
?
metrics
layer_metrics
layer_regularization_losses

layers
 non_trainable_variables
regularization_losses
	variables
trainable_variables
Y__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
?
!axis
	gamma
beta
"regularization_losses
#	variables
$trainable_variables
%	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"
_tf_keras_layer
?
&regularization_losses
'	variables
(trainable_variables
)	keras_api
*\&call_and_return_all_conditional_losses
]__call__"
_tf_keras_layer
?

kernel
	bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
*^&call_and_return_all_conditional_losses
___call__"
_tf_keras_layer
?


kernel
bias
.regularization_losses
/	variables
0trainable_variables
1	keras_api
*`&call_and_return_all_conditional_losses
a__call__"
_tf_keras_layer
?

kernel
bias
2regularization_losses
3	variables
4trainable_variables
5	keras_api
*b&call_and_return_all_conditional_losses
c__call__"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
6metrics
7layer_metrics
8layer_regularization_losses

9layers
:non_trainable_variables
"regularization_losses
#	variables
$trainable_variables
[__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
;metrics
<layer_metrics
=layer_regularization_losses

>layers
?non_trainable_variables
&regularization_losses
'	variables
(trainable_variables
]__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
?
@metrics
Alayer_metrics
Blayer_regularization_losses

Clayers
Dnon_trainable_variables
*regularization_losses
+	variables
,trainable_variables
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
?
Emetrics
Flayer_metrics
Glayer_regularization_losses

Hlayers
Inon_trainable_variables
.regularization_losses
/	variables
0trainable_variables
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Jmetrics
Klayer_metrics
Llayer_regularization_losses

Mlayers
Nnon_trainable_variables
2regularization_losses
3	variables
4trainable_variables
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
*__inference_polymorphic_action_fn_41279223
*__inference_polymorphic_action_fn_41279308?
???
FullArgSpec(
args ?
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults?
? 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_polymorphic_distribution_fn_41279368?
???
FullArgSpec(
args ?
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults?
? 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_get_initial_state_41279371?
???
FullArgSpec!
args?
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_<lambda>_41278821"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_<lambda>_41278818"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_41279104
0/discount0/observation0/reward0/step_type"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_41279116
batch_size"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_41279131"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_41279138"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec.
args&?#
jself
jinputs
jnetwork_state
varargs
 
varkwjkwargs
defaults?
? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec.
args&?#
jself
jinputs
jnetwork_state
varargs
 
varkwjkwargs
defaults?
? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 <
__inference_<lambda>_41278818?

? 
? "? 	5
__inference_<lambda>_41278821?

? 
? "? S
&__inference_get_initial_state_41279371)"?
?
?

batch_size 
? "? ?
*__inference_polymorphic_action_fn_41279223?	
???
???
???
TimeStep,
	step_type?
	step_type?????????&
reward?
reward?????????*
discount?
discount?????????9
observation*?'
observation??????????
? 
? "R?O

PolicyStep&
action?
action?????????
state? 
info? ?
*__inference_polymorphic_action_fn_41279308?	
???
???
???
TimeStep6
	step_type)?&
time_step/step_type?????????0
reward&?#
time_step/reward?????????4
discount(?%
time_step/discount?????????C
observation4?1
time_step/observation??????????
? 
? "R?O

PolicyStep&
action?
action?????????
state? 
info? ?
0__inference_polymorphic_distribution_fn_41279368?	
???
???
???
TimeStep,
	step_type?
	step_type?????????&
reward?
reward?????????*
discount?
discount?????????9
observation*?'
observation??????????
? 
? "???

PolicyStep?
action??????
`
B??

atol? 

loc??????????

rtol? 
J?G

allow_nan_statsp

namejDeterministic_1

validate_argsp 
?
j
parameters
? 
?
jnameEtf_agents.policies.greedy_policy.DeterministicWithLogProb_ACTTypeSpec
state? 
info? ?
&__inference_signature_wrapper_41279104?	
???
? 
???
.

0/discount ?

0/discount?????????
=
0/observation,?)
0/observation??????????
*
0/reward?
0/reward?????????
0
0/step_type!?
0/step_type?????????"+?(
&
action?
action?????????a
&__inference_signature_wrapper_4127911670?-
? 
&?#
!

batch_size?

batch_size "? Z
&__inference_signature_wrapper_412791310?

? 
? "?

int64?
int64 	>
&__inference_signature_wrapper_41279138?

? 
? "? 