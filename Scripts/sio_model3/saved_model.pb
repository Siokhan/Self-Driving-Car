ож
Ў§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
Й
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
executor_typestring ѕ
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeѕ"serve*2.1.02unknown8АЉ

ѓ
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:*
dtype0
ѓ
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:*
dtype0
ѓ
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђd*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	ђd*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:d*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:d2*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:2*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2
*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:2
*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:
*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:
*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
љ
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_3/kernel/m
Ѕ
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:*
dtype0
ђ
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:*
dtype0
љ
Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/m
Ѕ
*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*&
_output_shapes
:*
dtype0
ђ
Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/m
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes
:*
dtype0
љ
Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_5/kernel/m
Ѕ
*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*&
_output_shapes
:*
dtype0
ђ
Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/m
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes
:*
dtype0
Є
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђd*&
shared_nameAdam/dense_4/kernel/m
ђ
)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes
:	ђd*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:d*
dtype0
є
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

:d2*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:2*
dtype0
є
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2
*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes

:2
*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:
*
dtype0
є
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes

:
*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:*
dtype0
љ
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_3/kernel/v
Ѕ
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:*
dtype0
ђ
Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:*
dtype0
љ
Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/v
Ѕ
*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*&
_output_shapes
:*
dtype0
ђ
Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/v
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes
:*
dtype0
љ
Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_5/kernel/v
Ѕ
*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*&
_output_shapes
:*
dtype0
ђ
Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/v
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
:*
dtype0
Є
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђd*&
shared_nameAdam/dense_4/kernel/v
ђ
)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes
:	ђd*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:d*
dtype0
є
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

:d2*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:2*
dtype0
є
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2
*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes

:2
*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:
*
dtype0
є
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes

:
*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
­M
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ФM
valueАMBъM BЌM
М
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
R
$regularization_losses
%trainable_variables
&	variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
R
.regularization_losses
/trainable_variables
0	variables
1	keras_api
R
2regularization_losses
3trainable_variables
4	variables
5	keras_api
R
6regularization_losses
7trainable_variables
8	variables
9	keras_api
h

:kernel
;bias
<regularization_losses
=trainable_variables
>	variables
?	keras_api
h

@kernel
Abias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
h

Fkernel
Gbias
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
h

Lkernel
Mbias
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
п
Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratemІmїmЇmј(mЈ)mљ:mЉ;mњ@mЊAmћFmЋGmќLmЌMmўvЎvџvЏvю(vЮ)vъ:vЪ;vа@vАAvбFvБGvцLvЦMvд
 
f
0
1
2
3
(4
)5
:6
;7
@8
A9
F10
G11
L12
M13
f
0
1
2
3
(4
)5
:6
;7
@8
A9
F10
G11
L12
M13
џ

Wlayers
Xlayer_regularization_losses
regularization_losses
trainable_variables
Ynon_trainable_variables
	variables
Zmetrics
 
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
џ

[layers
regularization_losses
\layer_regularization_losses
trainable_variables
]non_trainable_variables
	variables
^metrics
 
 
 
џ

_layers
regularization_losses
`layer_regularization_losses
trainable_variables
anon_trainable_variables
	variables
bmetrics
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
џ

clayers
 regularization_losses
dlayer_regularization_losses
!trainable_variables
enon_trainable_variables
"	variables
fmetrics
 
 
 
џ

glayers
$regularization_losses
hlayer_regularization_losses
%trainable_variables
inon_trainable_variables
&	variables
jmetrics
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
џ

klayers
*regularization_losses
llayer_regularization_losses
+trainable_variables
mnon_trainable_variables
,	variables
nmetrics
 
 
 
џ

olayers
.regularization_losses
player_regularization_losses
/trainable_variables
qnon_trainable_variables
0	variables
rmetrics
 
 
 
џ

slayers
2regularization_losses
tlayer_regularization_losses
3trainable_variables
unon_trainable_variables
4	variables
vmetrics
 
 
 
џ

wlayers
6regularization_losses
xlayer_regularization_losses
7trainable_variables
ynon_trainable_variables
8	variables
zmetrics
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1

:0
;1
џ

{layers
<regularization_losses
|layer_regularization_losses
=trainable_variables
}non_trainable_variables
>	variables
~metrics
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

@0
A1

@0
A1
Ю

layers
Bregularization_losses
 ђlayer_regularization_losses
Ctrainable_variables
Ђnon_trainable_variables
D	variables
ѓmetrics
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

F0
G1

F0
G1
ъ
Ѓlayers
Hregularization_losses
 ёlayer_regularization_losses
Itrainable_variables
Ёnon_trainable_variables
J	variables
єmetrics
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

L0
M1
ъ
Єlayers
Nregularization_losses
 ѕlayer_regularization_losses
Otrainable_variables
Ѕnon_trainable_variables
P	variables
іmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
V
0
1
2
3
4
5
6
	7

8
9
10
11
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
 
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ћ
serving_default_conv2d_3_inputPlaceholder*1
_output_shapes
:         ї└*
dtype0*&
shape:         ї└
Ё
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_3_inputconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference_signature_wrapper_6203
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
№
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOpConst*<
Tin5
321	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*&
f!R
__inference__traced_save_6654
Й	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/v*;
Tin4
220*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__traced_restore_6807ГК
ыX
┬
__inference__traced_save_6654
file_prefix.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop
savev2_1_const

identity_1ѕбMergeV2CheckpointsбSaveV2бSaveV2_1Ц
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_69cd1afd99c442d3944271fca179840e/part2
StringJoin/inputs_1Ђ

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename■
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*љ
valueєBЃ/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesТ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЯ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *=
dtypes3
12/	2
SaveV2Ѓ
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardг
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1б
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesј
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices¤
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1с
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesг
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityЂ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Й
_input_shapesг
Е: :::::::	ђd:d:d2:2:2
:
:
:: : : : : :::::::	ђd:d:d2:2:2
:
:
::::::::	ђd:d:d2:2:2
:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
ь
Д
&__inference_dense_4_layer_call_fn_6436

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         d**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_59652
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
│
e
I__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_5847

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
└	
┌
A__inference_dense_4_layer_call_and_return_conditional_losses_5965

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:         d2
Eluќ
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
│
e
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_5814

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
У\
п
__inference__wrapped_model_5787
conv2d_3_input7
3sunny_model_conv2d_3_conv2d_readvariableop_resource8
4sunny_model_conv2d_3_biasadd_readvariableop_resource7
3sunny_model_conv2d_4_conv2d_readvariableop_resource8
4sunny_model_conv2d_4_biasadd_readvariableop_resource7
3sunny_model_conv2d_5_conv2d_readvariableop_resource8
4sunny_model_conv2d_5_biasadd_readvariableop_resource6
2sunny_model_dense_4_matmul_readvariableop_resource7
3sunny_model_dense_4_biasadd_readvariableop_resource6
2sunny_model_dense_5_matmul_readvariableop_resource7
3sunny_model_dense_5_biasadd_readvariableop_resource6
2sunny_model_dense_6_matmul_readvariableop_resource7
3sunny_model_dense_6_biasadd_readvariableop_resource6
2sunny_model_dense_7_matmul_readvariableop_resource7
3sunny_model_dense_7_biasadd_readvariableop_resource
identityѕб+Sunny_Model/conv2d_3/BiasAdd/ReadVariableOpб*Sunny_Model/conv2d_3/Conv2D/ReadVariableOpб+Sunny_Model/conv2d_4/BiasAdd/ReadVariableOpб*Sunny_Model/conv2d_4/Conv2D/ReadVariableOpб+Sunny_Model/conv2d_5/BiasAdd/ReadVariableOpб*Sunny_Model/conv2d_5/Conv2D/ReadVariableOpб*Sunny_Model/dense_4/BiasAdd/ReadVariableOpб)Sunny_Model/dense_4/MatMul/ReadVariableOpб*Sunny_Model/dense_5/BiasAdd/ReadVariableOpб)Sunny_Model/dense_5/MatMul/ReadVariableOpб*Sunny_Model/dense_6/BiasAdd/ReadVariableOpб)Sunny_Model/dense_6/MatMul/ReadVariableOpб*Sunny_Model/dense_7/BiasAdd/ReadVariableOpб)Sunny_Model/dense_7/MatMul/ReadVariableOpн
*Sunny_Model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3sunny_model_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*Sunny_Model/conv2d_3/Conv2D/ReadVariableOpВ
Sunny_Model/conv2d_3/Conv2DConv2Dconv2d_3_input2Sunny_Model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Dъ*
paddingVALID*
strides
2
Sunny_Model/conv2d_3/Conv2D╦
+Sunny_Model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4sunny_model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+Sunny_Model/conv2d_3/BiasAdd/ReadVariableOpП
Sunny_Model/conv2d_3/BiasAddBiasAdd$Sunny_Model/conv2d_3/Conv2D:output:03Sunny_Model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Dъ2
Sunny_Model/conv2d_3/BiasAddЮ
Sunny_Model/conv2d_3/EluElu%Sunny_Model/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:         Dъ2
Sunny_Model/conv2d_3/EluЖ
#Sunny_Model/max_pooling2d_3/MaxPoolMaxPool&Sunny_Model/conv2d_3/Elu:activations:0*/
_output_shapes
:         "O*
ksize
*
paddingVALID*
strides
2%
#Sunny_Model/max_pooling2d_3/MaxPoolн
*Sunny_Model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3sunny_model_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*Sunny_Model/conv2d_4/Conv2D/ReadVariableOpЅ
Sunny_Model/conv2d_4/Conv2DConv2D,Sunny_Model/max_pooling2d_3/MaxPool:output:02Sunny_Model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         &*
paddingVALID*
strides
2
Sunny_Model/conv2d_4/Conv2D╦
+Sunny_Model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4sunny_model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+Sunny_Model/conv2d_4/BiasAdd/ReadVariableOp▄
Sunny_Model/conv2d_4/BiasAddBiasAdd$Sunny_Model/conv2d_4/Conv2D:output:03Sunny_Model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         &2
Sunny_Model/conv2d_4/BiasAddю
Sunny_Model/conv2d_4/EluElu%Sunny_Model/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:         &2
Sunny_Model/conv2d_4/EluЖ
#Sunny_Model/max_pooling2d_4/MaxPoolMaxPool&Sunny_Model/conv2d_4/Elu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
2%
#Sunny_Model/max_pooling2d_4/MaxPoolн
*Sunny_Model/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3sunny_model_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*Sunny_Model/conv2d_5/Conv2D/ReadVariableOpЅ
Sunny_Model/conv2d_5/Conv2DConv2D,Sunny_Model/max_pooling2d_4/MaxPool:output:02Sunny_Model/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
Sunny_Model/conv2d_5/Conv2D╦
+Sunny_Model/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4sunny_model_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+Sunny_Model/conv2d_5/BiasAdd/ReadVariableOp▄
Sunny_Model/conv2d_5/BiasAddBiasAdd$Sunny_Model/conv2d_5/Conv2D:output:03Sunny_Model/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
Sunny_Model/conv2d_5/BiasAddю
Sunny_Model/conv2d_5/EluElu%Sunny_Model/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:         2
Sunny_Model/conv2d_5/EluЖ
#Sunny_Model/max_pooling2d_5/MaxPoolMaxPool&Sunny_Model/conv2d_5/Elu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
2%
#Sunny_Model/max_pooling2d_5/MaxPool┤
Sunny_Model/dropout_1/IdentityIdentity,Sunny_Model/max_pooling2d_5/MaxPool:output:0*
T0*/
_output_shapes
:         2 
Sunny_Model/dropout_1/IdentityІ
Sunny_Model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  2
Sunny_Model/flatten_1/Const╦
Sunny_Model/flatten_1/ReshapeReshape'Sunny_Model/dropout_1/Identity:output:0$Sunny_Model/flatten_1/Const:output:0*
T0*(
_output_shapes
:         ђ2
Sunny_Model/flatten_1/Reshape╩
)Sunny_Model/dense_4/MatMul/ReadVariableOpReadVariableOp2sunny_model_dense_4_matmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype02+
)Sunny_Model/dense_4/MatMul/ReadVariableOp¤
Sunny_Model/dense_4/MatMulMatMul&Sunny_Model/flatten_1/Reshape:output:01Sunny_Model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
Sunny_Model/dense_4/MatMul╚
*Sunny_Model/dense_4/BiasAdd/ReadVariableOpReadVariableOp3sunny_model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02,
*Sunny_Model/dense_4/BiasAdd/ReadVariableOpЛ
Sunny_Model/dense_4/BiasAddBiasAdd$Sunny_Model/dense_4/MatMul:product:02Sunny_Model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
Sunny_Model/dense_4/BiasAddЉ
Sunny_Model/dense_4/EluElu$Sunny_Model/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
Sunny_Model/dense_4/Elu╔
)Sunny_Model/dense_5/MatMul/ReadVariableOpReadVariableOp2sunny_model_dense_5_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype02+
)Sunny_Model/dense_5/MatMul/ReadVariableOp╬
Sunny_Model/dense_5/MatMulMatMul%Sunny_Model/dense_4/Elu:activations:01Sunny_Model/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
Sunny_Model/dense_5/MatMul╚
*Sunny_Model/dense_5/BiasAdd/ReadVariableOpReadVariableOp3sunny_model_dense_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02,
*Sunny_Model/dense_5/BiasAdd/ReadVariableOpЛ
Sunny_Model/dense_5/BiasAddBiasAdd$Sunny_Model/dense_5/MatMul:product:02Sunny_Model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
Sunny_Model/dense_5/BiasAddЉ
Sunny_Model/dense_5/EluElu$Sunny_Model/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:         22
Sunny_Model/dense_5/Elu╔
)Sunny_Model/dense_6/MatMul/ReadVariableOpReadVariableOp2sunny_model_dense_6_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02+
)Sunny_Model/dense_6/MatMul/ReadVariableOp╬
Sunny_Model/dense_6/MatMulMatMul%Sunny_Model/dense_5/Elu:activations:01Sunny_Model/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
Sunny_Model/dense_6/MatMul╚
*Sunny_Model/dense_6/BiasAdd/ReadVariableOpReadVariableOp3sunny_model_dense_6_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02,
*Sunny_Model/dense_6/BiasAdd/ReadVariableOpЛ
Sunny_Model/dense_6/BiasAddBiasAdd$Sunny_Model/dense_6/MatMul:product:02Sunny_Model/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
Sunny_Model/dense_6/BiasAddЉ
Sunny_Model/dense_6/EluElu$Sunny_Model/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
Sunny_Model/dense_6/Elu╔
)Sunny_Model/dense_7/MatMul/ReadVariableOpReadVariableOp2sunny_model_dense_7_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02+
)Sunny_Model/dense_7/MatMul/ReadVariableOp╬
Sunny_Model/dense_7/MatMulMatMul%Sunny_Model/dense_6/Elu:activations:01Sunny_Model/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Sunny_Model/dense_7/MatMul╚
*Sunny_Model/dense_7/BiasAdd/ReadVariableOpReadVariableOp3sunny_model_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*Sunny_Model/dense_7/BiasAdd/ReadVariableOpЛ
Sunny_Model/dense_7/BiasAddBiasAdd$Sunny_Model/dense_7/MatMul:product:02Sunny_Model/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Sunny_Model/dense_7/BiasAddь
IdentityIdentity$Sunny_Model/dense_7/BiasAdd:output:0,^Sunny_Model/conv2d_3/BiasAdd/ReadVariableOp+^Sunny_Model/conv2d_3/Conv2D/ReadVariableOp,^Sunny_Model/conv2d_4/BiasAdd/ReadVariableOp+^Sunny_Model/conv2d_4/Conv2D/ReadVariableOp,^Sunny_Model/conv2d_5/BiasAdd/ReadVariableOp+^Sunny_Model/conv2d_5/Conv2D/ReadVariableOp+^Sunny_Model/dense_4/BiasAdd/ReadVariableOp*^Sunny_Model/dense_4/MatMul/ReadVariableOp+^Sunny_Model/dense_5/BiasAdd/ReadVariableOp*^Sunny_Model/dense_5/MatMul/ReadVariableOp+^Sunny_Model/dense_6/BiasAdd/ReadVariableOp*^Sunny_Model/dense_6/MatMul/ReadVariableOp+^Sunny_Model/dense_7/BiasAdd/ReadVariableOp*^Sunny_Model/dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:         ї└::::::::::::::2Z
+Sunny_Model/conv2d_3/BiasAdd/ReadVariableOp+Sunny_Model/conv2d_3/BiasAdd/ReadVariableOp2X
*Sunny_Model/conv2d_3/Conv2D/ReadVariableOp*Sunny_Model/conv2d_3/Conv2D/ReadVariableOp2Z
+Sunny_Model/conv2d_4/BiasAdd/ReadVariableOp+Sunny_Model/conv2d_4/BiasAdd/ReadVariableOp2X
*Sunny_Model/conv2d_4/Conv2D/ReadVariableOp*Sunny_Model/conv2d_4/Conv2D/ReadVariableOp2Z
+Sunny_Model/conv2d_5/BiasAdd/ReadVariableOp+Sunny_Model/conv2d_5/BiasAdd/ReadVariableOp2X
*Sunny_Model/conv2d_5/Conv2D/ReadVariableOp*Sunny_Model/conv2d_5/Conv2D/ReadVariableOp2X
*Sunny_Model/dense_4/BiasAdd/ReadVariableOp*Sunny_Model/dense_4/BiasAdd/ReadVariableOp2V
)Sunny_Model/dense_4/MatMul/ReadVariableOp)Sunny_Model/dense_4/MatMul/ReadVariableOp2X
*Sunny_Model/dense_5/BiasAdd/ReadVariableOp*Sunny_Model/dense_5/BiasAdd/ReadVariableOp2V
)Sunny_Model/dense_5/MatMul/ReadVariableOp)Sunny_Model/dense_5/MatMul/ReadVariableOp2X
*Sunny_Model/dense_6/BiasAdd/ReadVariableOp*Sunny_Model/dense_6/BiasAdd/ReadVariableOp2V
)Sunny_Model/dense_6/MatMul/ReadVariableOp)Sunny_Model/dense_6/MatMul/ReadVariableOp2X
*Sunny_Model/dense_7/BiasAdd/ReadVariableOp*Sunny_Model/dense_7/BiasAdd/ReadVariableOp2V
)Sunny_Model/dense_7/MatMul/ReadVariableOp)Sunny_Model/dense_7/MatMul/ReadVariableOp:. *
(
_user_specified_nameconv2d_3_input
К
J
.__inference_max_pooling2d_4_layer_call_fn_5853

inputs
identityн
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4                                    **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_58472
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
У
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_5922

inputs
identityѕa
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/random_uniform/max╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformф
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub╚
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         2
dropout/random_uniform/mulХ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truedivЕ
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:         2
dropout/GreaterEqualx
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:         2
dropout/mulЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         2
dropout/Castѓ
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         2
dropout/mul_1m
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :& "
 
_user_specified_nameinputs
└	
┌
A__inference_dense_4_layer_call_and_return_conditional_losses_6429

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:         d2
Eluќ
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
╝
е
'__inference_conv2d_3_layer_call_fn_5808

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_58002
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╝
е
'__inference_conv2d_4_layer_call_fn_5841

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_58332
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
В
Д
&__inference_dense_7_layer_call_fn_6489

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_60332
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         
::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ж
D
(__inference_dropout_1_layer_call_fn_6407

inputs
identity│
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_59272
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :& "
 
_user_specified_nameinputs
Й	
┌
A__inference_dense_5_layer_call_and_return_conditional_losses_5988

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:         22
Eluќ
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*.
_input_shapes
:         d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
А7
╦
E__inference_Sunny_Model_layer_call_and_return_conditional_losses_6076
conv2d_3_input+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2*
&dense_5_statefulpartitionedcall_args_1*
&dense_5_statefulpartitionedcall_args_2*
&dense_6_statefulpartitionedcall_args_1*
&dense_6_statefulpartitionedcall_args_2*
&dense_7_statefulpartitionedcall_args_1*
&dense_7_statefulpartitionedcall_args_2
identityѕб conv2d_3/StatefulPartitionedCallб conv2d_4/StatefulPartitionedCallб conv2d_5/StatefulPartitionedCallбdense_4/StatefulPartitionedCallбdense_5/StatefulPartitionedCallбdense_6/StatefulPartitionedCallбdense_7/StatefulPartitionedCall╣
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallconv2d_3_input'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         Dъ**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_58002"
 conv2d_3/StatefulPartitionedCallЧ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         "O**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_58142!
max_pooling2d_3/PartitionedCallм
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         &**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_58332"
 conv2d_4/StatefulPartitionedCallЧ
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_58472!
max_pooling2d_4/PartitionedCallм
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_58662"
 conv2d_5/StatefulPartitionedCallЧ
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_58802!
max_pooling2d_5/PartitionedCallж
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_59272
dropout_1/PartitionedCall▄
flatten_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_59462
flatten_1/PartitionedCall┐
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         d**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_59652!
dense_4/StatefulPartitionedCall┼
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0&dense_5_statefulpartitionedcall_args_1&dense_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         2**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_59882!
dense_5/StatefulPartitionedCall┼
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0&dense_6_statefulpartitionedcall_args_1&dense_6_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_60112!
dense_6/StatefulPartitionedCall┼
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0&dense_7_statefulpartitionedcall_args_1&dense_7_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_60332!
dense_7/StatefulPartitionedCallь
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:         ї└::::::::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:. *
(
_user_specified_nameconv2d_3_input
я
Я
"__inference_signature_wrapper_6203
conv2d_3_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__wrapped_model_57872
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:         ї└::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_nameconv2d_3_input
У
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_6392

inputs
identityѕa
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/random_uniform/max╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformф
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub╚
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         2
dropout/random_uniform/mulХ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truedivЕ
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:         2
dropout/GreaterEqualx
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:         2
dropout/mulЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         2
dropout/Castѓ
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         2
dropout/mul_1m
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :& "
 
_user_specified_nameinputs
Й	
┌
A__inference_dense_6_layer_call_and_return_conditional_losses_6011

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:         
2
Eluќ
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*.
_input_shapes
:         2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
╝
е
'__inference_conv2d_5_layer_call_fn_5874

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_58662
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ш
a
(__inference_dropout_1_layer_call_fn_6402

inputs
identityѕбStatefulPartitionedCall╦
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_59222
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
К
J
.__inference_max_pooling2d_5_layer_call_fn_5886

inputs
identityн
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4                                    **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_58802
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
ї
У
*__inference_Sunny_Model_layer_call_fn_6126
conv2d_3_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_Sunny_Model_layer_call_and_return_conditional_losses_61092
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:         ї└::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_nameconv2d_3_input
Л8
№
E__inference_Sunny_Model_layer_call_and_return_conditional_losses_6046
conv2d_3_input+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2*
&dense_5_statefulpartitionedcall_args_1*
&dense_5_statefulpartitionedcall_args_2*
&dense_6_statefulpartitionedcall_args_1*
&dense_6_statefulpartitionedcall_args_2*
&dense_7_statefulpartitionedcall_args_1*
&dense_7_statefulpartitionedcall_args_2
identityѕб conv2d_3/StatefulPartitionedCallб conv2d_4/StatefulPartitionedCallб conv2d_5/StatefulPartitionedCallбdense_4/StatefulPartitionedCallбdense_5/StatefulPartitionedCallбdense_6/StatefulPartitionedCallбdense_7/StatefulPartitionedCallб!dropout_1/StatefulPartitionedCall╣
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallconv2d_3_input'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         Dъ**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_58002"
 conv2d_3/StatefulPartitionedCallЧ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         "O**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_58142!
max_pooling2d_3/PartitionedCallм
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         &**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_58332"
 conv2d_4/StatefulPartitionedCallЧ
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_58472!
max_pooling2d_4/PartitionedCallм
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_58662"
 conv2d_5/StatefulPartitionedCallЧ
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_58802!
max_pooling2d_5/PartitionedCallЂ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_59222#
!dropout_1/StatefulPartitionedCallС
flatten_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_59462
flatten_1/PartitionedCall┐
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         d**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_59652!
dense_4/StatefulPartitionedCall┼
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0&dense_5_statefulpartitionedcall_args_1&dense_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         2**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_59882!
dense_5/StatefulPartitionedCall┼
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0&dense_6_statefulpartitionedcall_args_1&dense_6_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_60112!
dense_6/StatefulPartitionedCall┼
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0&dense_7_statefulpartitionedcall_args_1&dense_7_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_60332!
dense_7/StatefulPartitionedCallЉ
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:         ї└::::::::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:. *
(
_user_specified_nameconv2d_3_input
▀I
д	
E__inference_Sunny_Model_layer_call_and_return_conditional_losses_6334

inputs+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identityѕбconv2d_3/BiasAdd/ReadVariableOpбconv2d_3/Conv2D/ReadVariableOpбconv2d_4/BiasAdd/ReadVariableOpбconv2d_4/Conv2D/ReadVariableOpбconv2d_5/BiasAdd/ReadVariableOpбconv2d_5/Conv2D/ReadVariableOpбdense_4/BiasAdd/ReadVariableOpбdense_4/MatMul/ReadVariableOpбdense_5/BiasAdd/ReadVariableOpбdense_5/MatMul/ReadVariableOpбdense_6/BiasAdd/ReadVariableOpбdense_6/MatMul/ReadVariableOpбdense_7/BiasAdd/ReadVariableOpбdense_7/MatMul/ReadVariableOp░
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_3/Conv2D/ReadVariableOp└
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Dъ*
paddingVALID*
strides
2
conv2d_3/Conv2DД
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpГ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Dъ2
conv2d_3/BiasAddy
conv2d_3/EluEluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:         Dъ2
conv2d_3/Eluк
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/Elu:activations:0*/
_output_shapes
:         "O*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool░
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOp┘
conv2d_4/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         &*
paddingVALID*
strides
2
conv2d_4/Conv2DД
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOpг
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         &2
conv2d_4/BiasAddx
conv2d_4/EluEluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:         &2
conv2d_4/Eluк
max_pooling2d_4/MaxPoolMaxPoolconv2d_4/Elu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool░
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOp┘
conv2d_5/Conv2DConv2D max_pooling2d_4/MaxPool:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
conv2d_5/Conv2DД
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOpг
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv2d_5/BiasAddx
conv2d_5/EluEluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:         2
conv2d_5/Eluк
max_pooling2d_5/MaxPoolMaxPoolconv2d_5/Elu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
2
max_pooling2d_5/MaxPoolљ
dropout_1/IdentityIdentity max_pooling2d_5/MaxPool:output:0*
T0*/
_output_shapes
:         2
dropout_1/Identitys
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  2
flatten_1/ConstЏ
flatten_1/ReshapeReshapedropout_1/Identity:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:         ђ2
flatten_1/Reshapeд
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype02
dense_4/MatMul/ReadVariableOpЪ
dense_4/MatMulMatMulflatten_1/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense_4/MatMulц
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02 
dense_4/BiasAdd/ReadVariableOpА
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense_4/BiasAddm
dense_4/EluEludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
dense_4/EluЦ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype02
dense_5/MatMul/ReadVariableOpъ
dense_5/MatMulMatMuldense_4/Elu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
dense_5/MatMulц
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
dense_5/BiasAdd/ReadVariableOpА
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
dense_5/BiasAddm
dense_5/EluEludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:         22
dense_5/EluЦ
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02
dense_6/MatMul/ReadVariableOpъ
dense_6/MatMulMatMuldense_5/Elu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_6/MatMulц
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_6/BiasAdd/ReadVariableOpА
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_6/BiasAddm
dense_6/EluEludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
dense_6/EluЦ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_7/MatMul/ReadVariableOpъ
dense_7/MatMulMatMuldense_6/Elu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_7/MatMulц
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOpА
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_7/BiasAdd╣
IdentityIdentitydense_7/BiasAdd:output:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:         ї└::::::::::::::2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
і
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_6413

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :& "
 
_user_specified_nameinputs
Й	
┌
A__inference_dense_5_layer_call_and_return_conditional_losses_6447

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:         22
Eluќ
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*.
_input_shapes
:         d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
З
Я
*__inference_Sunny_Model_layer_call_fn_6353

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identityѕбStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_Sunny_Model_layer_call_and_return_conditional_losses_61092
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:         ї└::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
с
█
B__inference_conv2d_5_layer_call_and_return_conditional_losses_5866

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpХ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpџ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddo
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2
Elu░
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
ђ└
╠
 __inference__traced_restore_6807
file_prefix$
 assignvariableop_conv2d_3_kernel$
 assignvariableop_1_conv2d_3_bias&
"assignvariableop_2_conv2d_4_kernel$
 assignvariableop_3_conv2d_4_bias&
"assignvariableop_4_conv2d_5_kernel$
 assignvariableop_5_conv2d_5_bias%
!assignvariableop_6_dense_4_kernel#
assignvariableop_7_dense_4_bias%
!assignvariableop_8_dense_5_kernel#
assignvariableop_9_dense_5_bias&
"assignvariableop_10_dense_6_kernel$
 assignvariableop_11_dense_6_bias&
"assignvariableop_12_dense_7_kernel$
 assignvariableop_13_dense_7_bias!
assignvariableop_14_adam_iter#
assignvariableop_15_adam_beta_1#
assignvariableop_16_adam_beta_2"
assignvariableop_17_adam_decay*
&assignvariableop_18_adam_learning_rate.
*assignvariableop_19_adam_conv2d_3_kernel_m,
(assignvariableop_20_adam_conv2d_3_bias_m.
*assignvariableop_21_adam_conv2d_4_kernel_m,
(assignvariableop_22_adam_conv2d_4_bias_m.
*assignvariableop_23_adam_conv2d_5_kernel_m,
(assignvariableop_24_adam_conv2d_5_bias_m-
)assignvariableop_25_adam_dense_4_kernel_m+
'assignvariableop_26_adam_dense_4_bias_m-
)assignvariableop_27_adam_dense_5_kernel_m+
'assignvariableop_28_adam_dense_5_bias_m-
)assignvariableop_29_adam_dense_6_kernel_m+
'assignvariableop_30_adam_dense_6_bias_m-
)assignvariableop_31_adam_dense_7_kernel_m+
'assignvariableop_32_adam_dense_7_bias_m.
*assignvariableop_33_adam_conv2d_3_kernel_v,
(assignvariableop_34_adam_conv2d_3_bias_v.
*assignvariableop_35_adam_conv2d_4_kernel_v,
(assignvariableop_36_adam_conv2d_4_bias_v.
*assignvariableop_37_adam_conv2d_5_kernel_v,
(assignvariableop_38_adam_conv2d_5_bias_v-
)assignvariableop_39_adam_dense_4_kernel_v+
'assignvariableop_40_adam_dense_4_bias_v-
)assignvariableop_41_adam_dense_5_kernel_v+
'assignvariableop_42_adam_dense_5_bias_v-
)assignvariableop_43_adam_dense_6_kernel_v+
'assignvariableop_44_adam_dense_6_bias_v-
)assignvariableop_45_adam_dense_7_kernel_v+
'assignvariableop_46_adam_dense_7_bias_v
identity_48ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9б	RestoreV2бRestoreV2_1ё
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*љ
valueєBЃ/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesВ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЎ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*м
_output_shapes┐
╝:::::::::::::::::::::::::::::::::::::::::::::::*=
dtypes3
12/	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identityљ
AssignVariableOpAssignVariableOp assignvariableop_conv2d_3_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1ќ
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_3_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2ў
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_4_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3ќ
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_4_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4ў
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_5_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5ќ
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_5_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ќ
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_4_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ћ
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_4_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8Ќ
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_5_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Ћ
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_5_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10Џ
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_6_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11Ў
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_6_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Џ
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_7_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13Ў
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_7_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0	*
_output_shapes
:2
Identity_14ќ
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15ў
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16ў
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17Ќ
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Ъ
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Б
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_conv2d_3_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20А
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_conv2d_3_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21Б
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_4_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22А
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_4_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23Б
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_5_kernel_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24А
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv2d_5_bias_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25б
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_4_kernel_mIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26а
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_4_bias_mIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27б
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_5_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28а
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_5_bias_mIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29б
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_6_kernel_mIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30а
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_6_bias_mIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31б
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_7_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32а
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_7_bias_mIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33Б
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_3_kernel_vIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34А
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv2d_3_bias_vIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35Б
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv2d_4_kernel_vIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36А
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv2d_4_bias_vIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37Б
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_5_kernel_vIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38А
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_5_bias_vIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39б
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_4_kernel_vIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40а
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_4_bias_vIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41б
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_5_kernel_vIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42а
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_5_bias_vIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43б
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_6_kernel_vIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44а
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_6_bias_vIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45б
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_7_kernel_vIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46а
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_7_bias_vIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46е
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesћ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpУ
Identity_47Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_47ш
Identity_48IdentityIdentity_47:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_48"#
identity_48Identity_48:output:0*М
_input_shapes┴
Й: :::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
і
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_5946

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :& "
 
_user_specified_nameinputs
х
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_5927

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         :& "
 
_user_specified_nameinputs
╣8
у
E__inference_Sunny_Model_layer_call_and_return_conditional_losses_6109

inputs+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2*
&dense_5_statefulpartitionedcall_args_1*
&dense_5_statefulpartitionedcall_args_2*
&dense_6_statefulpartitionedcall_args_1*
&dense_6_statefulpartitionedcall_args_2*
&dense_7_statefulpartitionedcall_args_1*
&dense_7_statefulpartitionedcall_args_2
identityѕб conv2d_3/StatefulPartitionedCallб conv2d_4/StatefulPartitionedCallб conv2d_5/StatefulPartitionedCallбdense_4/StatefulPartitionedCallбdense_5/StatefulPartitionedCallбdense_6/StatefulPartitionedCallбdense_7/StatefulPartitionedCallб!dropout_1/StatefulPartitionedCall▒
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputs'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         Dъ**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_58002"
 conv2d_3/StatefulPartitionedCallЧ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         "O**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_58142!
max_pooling2d_3/PartitionedCallм
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         &**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_58332"
 conv2d_4/StatefulPartitionedCallЧ
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_58472!
max_pooling2d_4/PartitionedCallм
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_58662"
 conv2d_5/StatefulPartitionedCallЧ
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_58802!
max_pooling2d_5/PartitionedCallЂ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_59222#
!dropout_1/StatefulPartitionedCallС
flatten_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_59462
flatten_1/PartitionedCall┐
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         d**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_59652!
dense_4/StatefulPartitionedCall┼
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0&dense_5_statefulpartitionedcall_args_1&dense_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         2**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_59882!
dense_5/StatefulPartitionedCall┼
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0&dense_6_statefulpartitionedcall_args_1&dense_6_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_60112!
dense_6/StatefulPartitionedCall┼
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0&dense_7_statefulpartitionedcall_args_1&dense_7_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_60332!
dense_7/StatefulPartitionedCallЉ
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:         ї└::::::::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
ї
У
*__inference_Sunny_Model_layer_call_fn_6175
conv2d_3_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_Sunny_Model_layer_call_and_return_conditional_losses_61582
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:         ї└::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_nameconv2d_3_input
█
D
(__inference_flatten_1_layer_call_fn_6418

inputs
identityг
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_59462
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :& "
 
_user_specified_nameinputs
З
Я
*__inference_Sunny_Model_layer_call_fn_6372

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identityѕбStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_Sunny_Model_layer_call_and_return_conditional_losses_61582
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:         ї└::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
К
J
.__inference_max_pooling2d_3_layer_call_fn_5820

inputs
identityн
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4                                    **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_58142
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
В
Д
&__inference_dense_5_layer_call_fn_6454

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         2**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_59882
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*.
_input_shapes
:         d::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
с
█
B__inference_conv2d_3_layer_call_and_return_conditional_losses_5800

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpХ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpџ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddo
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2
Elu░
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
с
█
B__inference_conv2d_4_layer_call_and_return_conditional_losses_5833

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpХ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpџ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddo
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2
Elu░
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
│
e
I__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_5880

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
Й	
┌
A__inference_dense_6_layer_call_and_return_conditional_losses_6465

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:         
2
Eluќ
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*.
_input_shapes
:         2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ѕ7
├
E__inference_Sunny_Model_layer_call_and_return_conditional_losses_6158

inputs+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2*
&dense_5_statefulpartitionedcall_args_1*
&dense_5_statefulpartitionedcall_args_2*
&dense_6_statefulpartitionedcall_args_1*
&dense_6_statefulpartitionedcall_args_2*
&dense_7_statefulpartitionedcall_args_1*
&dense_7_statefulpartitionedcall_args_2
identityѕб conv2d_3/StatefulPartitionedCallб conv2d_4/StatefulPartitionedCallб conv2d_5/StatefulPartitionedCallбdense_4/StatefulPartitionedCallбdense_5/StatefulPartitionedCallбdense_6/StatefulPartitionedCallбdense_7/StatefulPartitionedCall▒
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputs'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         Dъ**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_58002"
 conv2d_3/StatefulPartitionedCallЧ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         "O**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_58142!
max_pooling2d_3/PartitionedCallм
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         &**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_58332"
 conv2d_4/StatefulPartitionedCallЧ
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_58472!
max_pooling2d_4/PartitionedCallм
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_58662"
 conv2d_5/StatefulPartitionedCallЧ
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_58802!
max_pooling2d_5/PartitionedCallж
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_59272
dropout_1/PartitionedCall▄
flatten_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_59462
flatten_1/PartitionedCall┐
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         d**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_59652!
dense_4/StatefulPartitionedCall┼
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0&dense_5_statefulpartitionedcall_args_1&dense_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         2**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_59882!
dense_5/StatefulPartitionedCall┼
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0&dense_6_statefulpartitionedcall_args_1&dense_6_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_60112!
dense_6/StatefulPartitionedCall┼
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0&dense_7_statefulpartitionedcall_args_1&dense_7_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_60332!
dense_7/StatefulPartitionedCallь
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:         ї└::::::::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
Т
┌
A__inference_dense_7_layer_call_and_return_conditional_losses_6033

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
В
Д
&__inference_dense_6_layer_call_fn_6472

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_60112
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*.
_input_shapes
:         2::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Т
┌
A__inference_dense_7_layer_call_and_return_conditional_losses_6482

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ј^
д	
E__inference_Sunny_Model_layer_call_and_return_conditional_losses_6276

inputs+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identityѕбconv2d_3/BiasAdd/ReadVariableOpбconv2d_3/Conv2D/ReadVariableOpбconv2d_4/BiasAdd/ReadVariableOpбconv2d_4/Conv2D/ReadVariableOpбconv2d_5/BiasAdd/ReadVariableOpбconv2d_5/Conv2D/ReadVariableOpбdense_4/BiasAdd/ReadVariableOpбdense_4/MatMul/ReadVariableOpбdense_5/BiasAdd/ReadVariableOpбdense_5/MatMul/ReadVariableOpбdense_6/BiasAdd/ReadVariableOpбdense_6/MatMul/ReadVariableOpбdense_7/BiasAdd/ReadVariableOpбdense_7/MatMul/ReadVariableOp░
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_3/Conv2D/ReadVariableOp└
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Dъ*
paddingVALID*
strides
2
conv2d_3/Conv2DД
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpГ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Dъ2
conv2d_3/BiasAddy
conv2d_3/EluEluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:         Dъ2
conv2d_3/Eluк
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/Elu:activations:0*/
_output_shapes
:         "O*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool░
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOp┘
conv2d_4/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         &*
paddingVALID*
strides
2
conv2d_4/Conv2DД
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOpг
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         &2
conv2d_4/BiasAddx
conv2d_4/EluEluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:         &2
conv2d_4/Eluк
max_pooling2d_4/MaxPoolMaxPoolconv2d_4/Elu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool░
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOp┘
conv2d_5/Conv2DConv2D max_pooling2d_4/MaxPool:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
conv2d_5/Conv2DД
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOpг
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv2d_5/BiasAddx
conv2d_5/EluEluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:         2
conv2d_5/Eluк
max_pooling2d_5/MaxPoolMaxPoolconv2d_5/Elu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
2
max_pooling2d_5/MaxPoolu
dropout_1/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout_1/dropout/rateѓ
dropout_1/dropout/ShapeShape max_pooling2d_5/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/ShapeЉ
$dropout_1/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$dropout_1/dropout/random_uniform/minЉ
$dropout_1/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2&
$dropout_1/dropout/random_uniform/max┌
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:         *
dtype020
.dropout_1/dropout/random_uniform/RandomUniformм
$dropout_1/dropout/random_uniform/subSub-dropout_1/dropout/random_uniform/max:output:0-dropout_1/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2&
$dropout_1/dropout/random_uniform/sub­
$dropout_1/dropout/random_uniform/mulMul7dropout_1/dropout/random_uniform/RandomUniform:output:0(dropout_1/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         2&
$dropout_1/dropout/random_uniform/mulя
 dropout_1/dropout/random_uniformAdd(dropout_1/dropout/random_uniform/mul:z:0-dropout_1/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         2"
 dropout_1/dropout/random_uniformw
dropout_1/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout_1/dropout/sub/xЎ
dropout_1/dropout/subSub dropout_1/dropout/sub/x:output:0dropout_1/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_1/dropout/sub
dropout_1/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout_1/dropout/truediv/xБ
dropout_1/dropout/truedivRealDiv$dropout_1/dropout/truediv/x:output:0dropout_1/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_1/dropout/truedivЛ
dropout_1/dropout/GreaterEqualGreaterEqual$dropout_1/dropout/random_uniform:z:0dropout_1/dropout/rate:output:0*
T0*/
_output_shapes
:         2 
dropout_1/dropout/GreaterEqual░
dropout_1/dropout/mulMul max_pooling2d_5/MaxPool:output:0dropout_1/dropout/truediv:z:0*
T0*/
_output_shapes
:         2
dropout_1/dropout/mulЦ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         2
dropout_1/dropout/Castф
dropout_1/dropout/mul_1Muldropout_1/dropout/mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:         2
dropout_1/dropout/mul_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  2
flatten_1/ConstЏ
flatten_1/ReshapeReshapedropout_1/dropout/mul_1:z:0flatten_1/Const:output:0*
T0*(
_output_shapes
:         ђ2
flatten_1/Reshapeд
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype02
dense_4/MatMul/ReadVariableOpЪ
dense_4/MatMulMatMulflatten_1/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense_4/MatMulц
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02 
dense_4/BiasAdd/ReadVariableOpА
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense_4/BiasAddm
dense_4/EluEludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
dense_4/EluЦ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype02
dense_5/MatMul/ReadVariableOpъ
dense_5/MatMulMatMuldense_4/Elu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
dense_5/MatMulц
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
dense_5/BiasAdd/ReadVariableOpА
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
dense_5/BiasAddm
dense_5/EluEludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:         22
dense_5/EluЦ
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02
dense_6/MatMul/ReadVariableOpъ
dense_6/MatMulMatMuldense_5/Elu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_6/MatMulц
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_6/BiasAdd/ReadVariableOpА
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_6/BiasAddm
dense_6/EluEludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
dense_6/EluЦ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_7/MatMul/ReadVariableOpъ
dense_7/MatMulMatMuldense_6/Elu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_7/MatMulц
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOpА
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_7/BiasAdd╣
IdentityIdentitydense_7/BiasAdd:output:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:         ї└::::::::::::::2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
х
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_6397

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         :& "
 
_user_specified_nameinputs"»L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┬
serving_default«
S
conv2d_3_inputA
 serving_default_conv2d_3_input:0         ї└;
dense_70
StatefulPartitionedCall:0         tensorflow/serving/predict:ЮЛ
ЭM
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
Д_default_save_signature
е__call__
+Е&call_and_return_all_conditional_losses"╚I
_tf_keras_sequentialЕI{"class_name": "Sequential", "name": "Sunny_Model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Sunny_Model", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "batch_input_shape": [null, 140, 320, 3], "dtype": "float32", "filters": 24, "kernel_size": [5, 5], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": [5, 5], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 50, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 10, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "Sunny_Model", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "batch_input_shape": [null, 140, 320, 3], "dtype": "float32", "filters": 24, "kernel_size": [5, 5], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": [5, 5], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 50, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 10, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": [], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
┐"╝
_tf_keras_input_layerю{"class_name": "InputLayer", "name": "conv2d_3_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 140, 320, 3], "config": {"batch_input_shape": [null, 140, 320, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}}
д

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
ф__call__
+Ф&call_and_return_all_conditional_losses" 
_tf_keras_layerт{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 140, 320, 3], "config": {"name": "conv2d_3", "trainable": true, "batch_input_shape": [null, 140, 320, 3], "dtype": "float32", "filters": 24, "kernel_size": [5, 5], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
 
regularization_losses
trainable_variables
	variables
	keras_api
г__call__
+Г&call_and_return_all_conditional_losses"Ь
_tf_keras_layerн{"class_name": "MaxPooling2D", "name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ь

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
«__call__
+»&call_and_return_all_conditional_losses"К
_tf_keras_layerГ{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": [5, 5], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 24}}}}
 
$regularization_losses
%trainable_variables
&	variables
'	keras_api
░__call__
+▒&call_and_return_all_conditional_losses"Ь
_tf_keras_layerн{"class_name": "MaxPooling2D", "name": "max_pooling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ь

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
▓__call__
+│&call_and_return_all_conditional_losses"К
_tf_keras_layerГ{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 24}}}}
 
.regularization_losses
/trainable_variables
0	variables
1	keras_api
┤__call__
+х&call_and_return_all_conditional_losses"Ь
_tf_keras_layerн{"class_name": "MaxPooling2D", "name": "max_pooling2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
▒
2regularization_losses
3trainable_variables
4	variables
5	keras_api
Х__call__
+и&call_and_return_all_conditional_losses"а
_tf_keras_layerє{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
▓
6regularization_losses
7trainable_variables
8	variables
9	keras_api
И__call__
+╣&call_and_return_all_conditional_losses"А
_tf_keras_layerЄ{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
З

:kernel
;bias
<regularization_losses
=trainable_variables
>	variables
?	keras_api
║__call__
+╗&call_and_return_all_conditional_losses"═
_tf_keras_layer│{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 100, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 384}}}}
з

@kernel
Abias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
╝__call__
+й&call_and_return_all_conditional_losses"╠
_tf_keras_layer▓{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 50, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}}
Ы

Fkernel
Gbias
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
Й__call__
+┐&call_and_return_all_conditional_losses"╦
_tf_keras_layer▒{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 10, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}}
З

Lkernel
Mbias
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
└__call__
+┴&call_and_return_all_conditional_losses"═
_tf_keras_layer│{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}}
в
Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratemІmїmЇmј(mЈ)mљ:mЉ;mњ@mЊAmћFmЋGmќLmЌMmўvЎvџvЏvю(vЮ)vъ:vЪ;vа@vАAvбFvБGvцLvЦMvд"
	optimizer
 "
trackable_list_wrapper
є
0
1
2
3
(4
)5
:6
;7
@8
A9
F10
G11
L12
M13"
trackable_list_wrapper
є
0
1
2
3
(4
)5
:6
;7
@8
A9
F10
G11
L12
M13"
trackable_list_wrapper
╗

Wlayers
Xlayer_regularization_losses
regularization_losses
trainable_variables
Ynon_trainable_variables
	variables
Zmetrics
е__call__
Д_default_save_signature
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
-
┬serving_default"
signature_map
):'2conv2d_3/kernel
:2conv2d_3/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Ю

[layers
regularization_losses
\layer_regularization_losses
trainable_variables
]non_trainable_variables
	variables
^metrics
ф__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ю

_layers
regularization_losses
`layer_regularization_losses
trainable_variables
anon_trainable_variables
	variables
bmetrics
г__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_4/kernel
:2conv2d_4/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Ю

clayers
 regularization_losses
dlayer_regularization_losses
!trainable_variables
enon_trainable_variables
"	variables
fmetrics
«__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ю

glayers
$regularization_losses
hlayer_regularization_losses
%trainable_variables
inon_trainable_variables
&	variables
jmetrics
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_5/kernel
:2conv2d_5/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
Ю

klayers
*regularization_losses
llayer_regularization_losses
+trainable_variables
mnon_trainable_variables
,	variables
nmetrics
▓__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ю

olayers
.regularization_losses
player_regularization_losses
/trainable_variables
qnon_trainable_variables
0	variables
rmetrics
┤__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ю

slayers
2regularization_losses
tlayer_regularization_losses
3trainable_variables
unon_trainable_variables
4	variables
vmetrics
Х__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ю

wlayers
6regularization_losses
xlayer_regularization_losses
7trainable_variables
ynon_trainable_variables
8	variables
zmetrics
И__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses"
_generic_user_object
!:	ђd2dense_4/kernel
:d2dense_4/bias
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
Ю

{layers
<regularization_losses
|layer_regularization_losses
=trainable_variables
}non_trainable_variables
>	variables
~metrics
║__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses"
_generic_user_object
 :d22dense_5/kernel
:22dense_5/bias
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
а

layers
Bregularization_losses
 ђlayer_regularization_losses
Ctrainable_variables
Ђnon_trainable_variables
D	variables
ѓmetrics
╝__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
 :2
2dense_6/kernel
:
2dense_6/bias
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
А
Ѓlayers
Hregularization_losses
 ёlayer_regularization_losses
Itrainable_variables
Ёnon_trainable_variables
J	variables
єmetrics
Й__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
 :
2dense_7/kernel
:2dense_7/bias
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
А
Єlayers
Nregularization_losses
 ѕlayer_regularization_losses
Otrainable_variables
Ѕnon_trainable_variables
P	variables
іmetrics
└__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
v
0
1
2
3
4
5
6
	7

8
9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.:,2Adam/conv2d_3/kernel/m
 :2Adam/conv2d_3/bias/m
.:,2Adam/conv2d_4/kernel/m
 :2Adam/conv2d_4/bias/m
.:,2Adam/conv2d_5/kernel/m
 :2Adam/conv2d_5/bias/m
&:$	ђd2Adam/dense_4/kernel/m
:d2Adam/dense_4/bias/m
%:#d22Adam/dense_5/kernel/m
:22Adam/dense_5/bias/m
%:#2
2Adam/dense_6/kernel/m
:
2Adam/dense_6/bias/m
%:#
2Adam/dense_7/kernel/m
:2Adam/dense_7/bias/m
.:,2Adam/conv2d_3/kernel/v
 :2Adam/conv2d_3/bias/v
.:,2Adam/conv2d_4/kernel/v
 :2Adam/conv2d_4/bias/v
.:,2Adam/conv2d_5/kernel/v
 :2Adam/conv2d_5/bias/v
&:$	ђd2Adam/dense_4/kernel/v
:d2Adam/dense_4/bias/v
%:#d22Adam/dense_5/kernel/v
:22Adam/dense_5/bias/v
%:#2
2Adam/dense_6/kernel/v
:
2Adam/dense_6/bias/v
%:#
2Adam/dense_7/kernel/v
:2Adam/dense_7/bias/v
Ь2в
__inference__wrapped_model_5787К
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/
conv2d_3_input         ї└
Ш2з
*__inference_Sunny_Model_layer_call_fn_6126
*__inference_Sunny_Model_layer_call_fn_6372
*__inference_Sunny_Model_layer_call_fn_6353
*__inference_Sunny_Model_layer_call_fn_6175└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Р2▀
E__inference_Sunny_Model_layer_call_and_return_conditional_losses_6334
E__inference_Sunny_Model_layer_call_and_return_conditional_losses_6276
E__inference_Sunny_Model_layer_call_and_return_conditional_losses_6046
E__inference_Sunny_Model_layer_call_and_return_conditional_losses_6076└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
є2Ѓ
'__inference_conv2d_3_layer_call_fn_5808О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           
А2ъ
B__inference_conv2d_3_layer_call_and_return_conditional_losses_5800О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           
ќ2Њ
.__inference_max_pooling2d_3_layer_call_fn_5820Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
▒2«
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_5814Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
є2Ѓ
'__inference_conv2d_4_layer_call_fn_5841О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           
А2ъ
B__inference_conv2d_4_layer_call_and_return_conditional_losses_5833О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           
ќ2Њ
.__inference_max_pooling2d_4_layer_call_fn_5853Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
▒2«
I__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_5847Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
є2Ѓ
'__inference_conv2d_5_layer_call_fn_5874О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           
А2ъ
B__inference_conv2d_5_layer_call_and_return_conditional_losses_5866О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           
ќ2Њ
.__inference_max_pooling2d_5_layer_call_fn_5886Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
▒2«
I__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_5880Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
ј2І
(__inference_dropout_1_layer_call_fn_6402
(__inference_dropout_1_layer_call_fn_6407┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
─2┴
C__inference_dropout_1_layer_call_and_return_conditional_losses_6397
C__inference_dropout_1_layer_call_and_return_conditional_losses_6392┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
м2¤
(__inference_flatten_1_layer_call_fn_6418б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_flatten_1_layer_call_and_return_conditional_losses_6413б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
л2═
&__inference_dense_4_layer_call_fn_6436б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
в2У
A__inference_dense_4_layer_call_and_return_conditional_losses_6429б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
л2═
&__inference_dense_5_layer_call_fn_6454б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
в2У
A__inference_dense_5_layer_call_and_return_conditional_losses_6447б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
л2═
&__inference_dense_6_layer_call_fn_6472б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
в2У
A__inference_dense_6_layer_call_and_return_conditional_losses_6465б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
л2═
&__inference_dense_7_layer_call_fn_6489б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
в2У
A__inference_dense_7_layer_call_and_return_conditional_losses_6482б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
8B6
"__inference_signature_wrapper_6203conv2d_3_input╠
E__inference_Sunny_Model_layer_call_and_return_conditional_losses_6046ѓ():;@AFGLMIбF
?б<
2і/
conv2d_3_input         ї└
p

 
ф "%б"
і
0         
џ ╠
E__inference_Sunny_Model_layer_call_and_return_conditional_losses_6076ѓ():;@AFGLMIбF
?б<
2і/
conv2d_3_input         ї└
p 

 
ф "%б"
і
0         
џ ├
E__inference_Sunny_Model_layer_call_and_return_conditional_losses_6276z():;@AFGLMAб>
7б4
*і'
inputs         ї└
p

 
ф "%б"
і
0         
џ ├
E__inference_Sunny_Model_layer_call_and_return_conditional_losses_6334z():;@AFGLMAб>
7б4
*і'
inputs         ї└
p 

 
ф "%б"
і
0         
џ Б
*__inference_Sunny_Model_layer_call_fn_6126u():;@AFGLMIбF
?б<
2і/
conv2d_3_input         ї└
p

 
ф "і         Б
*__inference_Sunny_Model_layer_call_fn_6175u():;@AFGLMIбF
?б<
2і/
conv2d_3_input         ї└
p 

 
ф "і         Џ
*__inference_Sunny_Model_layer_call_fn_6353m():;@AFGLMAб>
7б4
*і'
inputs         ї└
p

 
ф "і         Џ
*__inference_Sunny_Model_layer_call_fn_6372m():;@AFGLMAб>
7б4
*і'
inputs         ї└
p 

 
ф "і         ф
__inference__wrapped_model_5787є():;@AFGLMAб>
7б4
2і/
conv2d_3_input         ї└
ф "1ф.
,
dense_7!і
dense_7         О
B__inference_conv2d_3_layer_call_and_return_conditional_losses_5800љIбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                           
џ »
'__inference_conv2d_3_layer_call_fn_5808ЃIбF
?б<
:і7
inputs+                           
ф "2і/+                           О
B__inference_conv2d_4_layer_call_and_return_conditional_losses_5833љIбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                           
џ »
'__inference_conv2d_4_layer_call_fn_5841ЃIбF
?б<
:і7
inputs+                           
ф "2і/+                           О
B__inference_conv2d_5_layer_call_and_return_conditional_losses_5866љ()IбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                           
џ »
'__inference_conv2d_5_layer_call_fn_5874Ѓ()IбF
?б<
:і7
inputs+                           
ф "2і/+                           б
A__inference_dense_4_layer_call_and_return_conditional_losses_6429]:;0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         d
џ z
&__inference_dense_4_layer_call_fn_6436P:;0б-
&б#
!і
inputs         ђ
ф "і         dА
A__inference_dense_5_layer_call_and_return_conditional_losses_6447\@A/б,
%б"
 і
inputs         d
ф "%б"
і
0         2
џ y
&__inference_dense_5_layer_call_fn_6454O@A/б,
%б"
 і
inputs         d
ф "і         2А
A__inference_dense_6_layer_call_and_return_conditional_losses_6465\FG/б,
%б"
 і
inputs         2
ф "%б"
і
0         

џ y
&__inference_dense_6_layer_call_fn_6472OFG/б,
%б"
 і
inputs         2
ф "і         
А
A__inference_dense_7_layer_call_and_return_conditional_losses_6482\LM/б,
%б"
 і
inputs         

ф "%б"
і
0         
џ y
&__inference_dense_7_layer_call_fn_6489OLM/б,
%б"
 і
inputs         

ф "і         │
C__inference_dropout_1_layer_call_and_return_conditional_losses_6392l;б8
1б.
(і%
inputs         
p
ф "-б*
#і 
0         
џ │
C__inference_dropout_1_layer_call_and_return_conditional_losses_6397l;б8
1б.
(і%
inputs         
p 
ф "-б*
#і 
0         
џ І
(__inference_dropout_1_layer_call_fn_6402_;б8
1б.
(і%
inputs         
p
ф " і         І
(__inference_dropout_1_layer_call_fn_6407_;б8
1б.
(і%
inputs         
p 
ф " і         е
C__inference_flatten_1_layer_call_and_return_conditional_losses_6413a7б4
-б*
(і%
inputs         
ф "&б#
і
0         ђ
џ ђ
(__inference_flatten_1_layer_call_fn_6418T7б4
-б*
(і%
inputs         
ф "і         ђВ
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_5814ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ─
.__inference_max_pooling2d_3_layer_call_fn_5820ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    В
I__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_5847ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ─
.__inference_max_pooling2d_4_layer_call_fn_5853ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    В
I__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_5880ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ─
.__inference_max_pooling2d_5_layer_call_fn_5886ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ┐
"__inference_signature_wrapper_6203ў():;@AFGLMSбP
б 
IфF
D
conv2d_3_input2і/
conv2d_3_input         ї└"1ф.
,
dense_7!і
dense_7         