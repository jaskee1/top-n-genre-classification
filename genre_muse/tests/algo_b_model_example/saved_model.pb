
Ú
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

ú
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
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
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Í

conv2d_167/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_167/kernel

%conv2d_167/kernel/Read/ReadVariableOpReadVariableOpconv2d_167/kernel*&
_output_shapes
:*
dtype0
v
conv2d_167/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_167/bias
o
#conv2d_167/bias/Read/ReadVariableOpReadVariableOpconv2d_167/bias*
_output_shapes
:*
dtype0

batch_normalization_158/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_158/gamma

1batch_normalization_158/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_158/gamma*
_output_shapes
:*
dtype0

batch_normalization_158/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_158/beta

0batch_normalization_158/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_158/beta*
_output_shapes
:*
dtype0

#batch_normalization_158/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_158/moving_mean

7batch_normalization_158/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_158/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_158/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_158/moving_variance

;batch_normalization_158/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_158/moving_variance*
_output_shapes
:*
dtype0

conv2d_168/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_168/kernel

%conv2d_168/kernel/Read/ReadVariableOpReadVariableOpconv2d_168/kernel*&
_output_shapes
:*
dtype0
v
conv2d_168/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_168/bias
o
#conv2d_168/bias/Read/ReadVariableOpReadVariableOpconv2d_168/bias*
_output_shapes
:*
dtype0

batch_normalization_159/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_159/gamma

1batch_normalization_159/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_159/gamma*
_output_shapes
:*
dtype0

batch_normalization_159/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_159/beta

0batch_normalization_159/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_159/beta*
_output_shapes
:*
dtype0

#batch_normalization_159/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_159/moving_mean

7batch_normalization_159/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_159/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_159/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_159/moving_variance

;batch_normalization_159/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_159/moving_variance*
_output_shapes
:*
dtype0

conv2d_169/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_169/kernel

%conv2d_169/kernel/Read/ReadVariableOpReadVariableOpconv2d_169/kernel*&
_output_shapes
: *
dtype0
v
conv2d_169/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_169/bias
o
#conv2d_169/bias/Read/ReadVariableOpReadVariableOpconv2d_169/bias*
_output_shapes
: *
dtype0

batch_normalization_160/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_160/gamma

1batch_normalization_160/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_160/gamma*
_output_shapes
: *
dtype0

batch_normalization_160/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_160/beta

0batch_normalization_160/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_160/beta*
_output_shapes
: *
dtype0

#batch_normalization_160/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization_160/moving_mean

7batch_normalization_160/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_160/moving_mean*
_output_shapes
: *
dtype0
¦
'batch_normalization_160/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'batch_normalization_160/moving_variance

;batch_normalization_160/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_160/moving_variance*
_output_shapes
: *
dtype0

conv2d_170/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*"
shared_nameconv2d_170/kernel

%conv2d_170/kernel/Read/ReadVariableOpReadVariableOpconv2d_170/kernel*&
_output_shapes
: @*
dtype0
v
conv2d_170/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_170/bias
o
#conv2d_170/bias/Read/ReadVariableOpReadVariableOpconv2d_170/bias*
_output_shapes
:@*
dtype0

batch_normalization_161/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatch_normalization_161/gamma

1batch_normalization_161/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_161/gamma*
_output_shapes
:@*
dtype0

batch_normalization_161/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_161/beta

0batch_normalization_161/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_161/beta*
_output_shapes
:@*
dtype0

#batch_normalization_161/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization_161/moving_mean

7batch_normalization_161/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_161/moving_mean*
_output_shapes
:@*
dtype0
¦
'batch_normalization_161/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'batch_normalization_161/moving_variance

;batch_normalization_161/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_161/moving_variance*
_output_shapes
:@*
dtype0

conv2d_171/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameconv2d_171/kernel

%conv2d_171/kernel/Read/ReadVariableOpReadVariableOpconv2d_171/kernel*'
_output_shapes
:@*
dtype0
w
conv2d_171/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_171/bias
p
#conv2d_171/bias/Read/ReadVariableOpReadVariableOpconv2d_171/bias*
_output_shapes	
:*
dtype0

batch_normalization_162/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_162/gamma

1batch_normalization_162/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_162/gamma*
_output_shapes	
:*
dtype0

batch_normalization_162/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_162/beta

0batch_normalization_162/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_162/beta*
_output_shapes	
:*
dtype0

#batch_normalization_162/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_162/moving_mean

7batch_normalization_162/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_162/moving_mean*
_output_shapes	
:*
dtype0
§
'batch_normalization_162/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_162/moving_variance
 
;batch_normalization_162/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_162/moving_variance*
_output_shapes	
:*
dtype0
{
dense_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	M
* 
shared_namedense_37/kernel
t
#dense_37/kernel/Read/ReadVariableOpReadVariableOpdense_37/kernel*
_output_shapes
:	M
*
dtype0
r
dense_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_37/bias
k
!dense_37/bias/Read/ReadVariableOpReadVariableOpdense_37/bias*
_output_shapes
:
*
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
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv2d_167/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_167/kernel/m

,Adam/conv2d_167/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_167/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_167/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_167/bias/m
}
*Adam/conv2d_167/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_167/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_158/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_158/gamma/m

8Adam/batch_normalization_158/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_158/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_158/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_158/beta/m

7Adam/batch_normalization_158/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_158/beta/m*
_output_shapes
:*
dtype0

Adam/conv2d_168/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_168/kernel/m

,Adam/conv2d_168/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_168/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_168/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_168/bias/m
}
*Adam/conv2d_168/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_168/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_159/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_159/gamma/m

8Adam/batch_normalization_159/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_159/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_159/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_159/beta/m

7Adam/batch_normalization_159/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_159/beta/m*
_output_shapes
:*
dtype0

Adam/conv2d_169/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_169/kernel/m

,Adam/conv2d_169/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_169/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv2d_169/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_169/bias/m
}
*Adam/conv2d_169/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_169/bias/m*
_output_shapes
: *
dtype0
 
$Adam/batch_normalization_160/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adam/batch_normalization_160/gamma/m

8Adam/batch_normalization_160/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_160/gamma/m*
_output_shapes
: *
dtype0

#Adam/batch_normalization_160/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_160/beta/m

7Adam/batch_normalization_160/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_160/beta/m*
_output_shapes
: *
dtype0

Adam/conv2d_170/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/conv2d_170/kernel/m

,Adam/conv2d_170/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_170/kernel/m*&
_output_shapes
: @*
dtype0

Adam/conv2d_170/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_170/bias/m
}
*Adam/conv2d_170/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_170/bias/m*
_output_shapes
:@*
dtype0
 
$Adam/batch_normalization_161/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$Adam/batch_normalization_161/gamma/m

8Adam/batch_normalization_161/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_161/gamma/m*
_output_shapes
:@*
dtype0

#Adam/batch_normalization_161/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_161/beta/m

7Adam/batch_normalization_161/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_161/beta/m*
_output_shapes
:@*
dtype0

Adam/conv2d_171/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/conv2d_171/kernel/m

,Adam/conv2d_171/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_171/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv2d_171/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_171/bias/m
~
*Adam/conv2d_171/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_171/bias/m*
_output_shapes	
:*
dtype0
¡
$Adam/batch_normalization_162/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_162/gamma/m

8Adam/batch_normalization_162/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_162/gamma/m*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_162/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_162/beta/m

7Adam/batch_normalization_162/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_162/beta/m*
_output_shapes	
:*
dtype0

Adam/dense_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	M
*'
shared_nameAdam/dense_37/kernel/m

*Adam/dense_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/m*
_output_shapes
:	M
*
dtype0

Adam/dense_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_37/bias/m
y
(Adam/dense_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/m*
_output_shapes
:
*
dtype0

Adam/conv2d_167/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_167/kernel/v

,Adam/conv2d_167/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_167/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_167/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_167/bias/v
}
*Adam/conv2d_167/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_167/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_158/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_158/gamma/v

8Adam/batch_normalization_158/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_158/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_158/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_158/beta/v

7Adam/batch_normalization_158/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_158/beta/v*
_output_shapes
:*
dtype0

Adam/conv2d_168/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_168/kernel/v

,Adam/conv2d_168/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_168/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_168/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_168/bias/v
}
*Adam/conv2d_168/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_168/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_159/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_159/gamma/v

8Adam/batch_normalization_159/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_159/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_159/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_159/beta/v

7Adam/batch_normalization_159/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_159/beta/v*
_output_shapes
:*
dtype0

Adam/conv2d_169/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_169/kernel/v

,Adam/conv2d_169/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_169/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_169/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_169/bias/v
}
*Adam/conv2d_169/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_169/bias/v*
_output_shapes
: *
dtype0
 
$Adam/batch_normalization_160/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adam/batch_normalization_160/gamma/v

8Adam/batch_normalization_160/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_160/gamma/v*
_output_shapes
: *
dtype0

#Adam/batch_normalization_160/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_160/beta/v

7Adam/batch_normalization_160/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_160/beta/v*
_output_shapes
: *
dtype0

Adam/conv2d_170/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/conv2d_170/kernel/v

,Adam/conv2d_170/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_170/kernel/v*&
_output_shapes
: @*
dtype0

Adam/conv2d_170/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_170/bias/v
}
*Adam/conv2d_170/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_170/bias/v*
_output_shapes
:@*
dtype0
 
$Adam/batch_normalization_161/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$Adam/batch_normalization_161/gamma/v

8Adam/batch_normalization_161/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_161/gamma/v*
_output_shapes
:@*
dtype0

#Adam/batch_normalization_161/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_161/beta/v

7Adam/batch_normalization_161/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_161/beta/v*
_output_shapes
:@*
dtype0

Adam/conv2d_171/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/conv2d_171/kernel/v

,Adam/conv2d_171/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_171/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv2d_171/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_171/bias/v
~
*Adam/conv2d_171/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_171/bias/v*
_output_shapes	
:*
dtype0
¡
$Adam/batch_normalization_162/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_162/gamma/v

8Adam/batch_normalization_162/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_162/gamma/v*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_162/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_162/beta/v

7Adam/batch_normalization_162/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_162/beta/v*
_output_shapes	
:*
dtype0

Adam/dense_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	M
*'
shared_nameAdam/dense_37/kernel/v

*Adam/dense_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/v*
_output_shapes
:	M
*
dtype0

Adam/dense_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_37/bias/v
y
(Adam/dense_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
²°
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ì¯
valueá¯BÝ¯ BÕ¯
Ý
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer-15
layer-16
layer_with_weights-10
layer-17
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
¦

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*
Õ
$axis
	%gamma
&beta
'moving_mean
(moving_variance
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*

/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses* 
¦

5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses*
Õ
=axis
	>gamma
?beta
@moving_mean
Amoving_variance
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses*

H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses* 
¦

Nkernel
Obias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses*
Õ
Vaxis
	Wgamma
Xbeta
Ymoving_mean
Zmoving_variance
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses*

a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses* 
¦

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses*
Õ
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses*

z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬
	variables
 trainable_variables
¡regularization_losses
¢	keras_api
£_random_generator
¤__call__
+¥&call_and_return_all_conditional_losses* 
®
¦kernel
	§bias
¨	variables
©trainable_variables
ªregularization_losses
«	keras_api
¬__call__
+­&call_and_return_all_conditional_losses*

	®iter
¯beta_1
°beta_2

±decay
²learning_ratemm%m &m¡5m¢6m£>m¤?m¥Nm¦Om§Wm¨Xm©gmªhm«pm¬qm­	m®	m¯	m°	m±	¦m²	§m³v´vµ%v¶&v·5v¸6v¹>vº?v»Nv¼Ov½Wv¾Xv¿gvÀhvÁpvÂqvÃ	vÄ	vÅ	vÆ	vÇ	¦vÈ	§vÉ*

0
1
%2
&3
'4
(5
56
67
>8
?9
@10
A11
N12
O13
W14
X15
Y16
Z17
g18
h19
p20
q21
r22
s23
24
25
26
27
28
29
¦30
§31*
°
0
1
%2
&3
54
65
>6
?7
N8
O9
W10
X11
g12
h13
p14
q15
16
17
18
19
¦20
§21*
* 
µ
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

¸serving_default* 
a[
VARIABLE_VALUEconv2d_167/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_167/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_158/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_158/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_158/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_158/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
%0
&1
'2
(3*

%0
&1*
* 

¾non_trainable_variables
¿layers
Àmetrics
 Álayer_regularization_losses
Âlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEconv2d_168/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_168/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

50
61*

50
61*
* 

Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_159/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_159/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_159/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_159/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
>0
?1
@2
A3*

>0
?1*
* 

Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEconv2d_169/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_169/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

N0
O1*

N0
O1*
* 

×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_160/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_160/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_160/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_160/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
W0
X1
Y2
Z3*

W0
X1*
* 

Ünon_trainable_variables
Ýlayers
Þmetrics
 ßlayer_regularization_losses
àlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEconv2d_170/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_170/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

g0
h1*

g0
h1*
* 

ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_161/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_161/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_161/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_161/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
p0
q1
r2
s3*

p0
q1*
* 

ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEconv2d_171/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_171/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_162/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_162/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_162/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_162/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

0
1*
* 

únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
 trainable_variables
¡regularization_losses
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses* 
* 
* 
* 
`Z
VARIABLE_VALUEdense_37/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_37/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*

¦0
§1*

¦0
§1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¨	variables
©trainable_variables
ªregularization_losses
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
L
'0
(1
@2
A3
Y4
Z5
r6
s7
8
9*

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17*

0
1*
* 
* 
* 
* 
* 
* 
* 
* 

'0
(1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

@0
A1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Y0
Z1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

r0
s1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

total

count
	variables
	keras_api*
M

total

count

_fn_kwargs
	variables
	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

	variables*
~
VARIABLE_VALUEAdam/conv2d_167/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_167/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_158/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_158/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_168/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_168/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_159/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_159/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_169/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_169/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_160/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_160/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_170/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_170/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_161/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_161/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_171/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_171/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_162/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_162/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_37/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_37/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_167/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_167/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_158/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_158/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_168/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_168/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_159/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_159/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_169/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_169/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_160/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_160/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_170/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_170/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_161/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_161/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_171/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d_171/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_162/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_162/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_37/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_37/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

 serving_default_conv2d_167_inputPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ °


StatefulPartitionedCallStatefulPartitionedCall serving_default_conv2d_167_inputconv2d_167/kernelconv2d_167/biasbatch_normalization_158/gammabatch_normalization_158/beta#batch_normalization_158/moving_mean'batch_normalization_158/moving_varianceconv2d_168/kernelconv2d_168/biasbatch_normalization_159/gammabatch_normalization_159/beta#batch_normalization_159/moving_mean'batch_normalization_159/moving_varianceconv2d_169/kernelconv2d_169/biasbatch_normalization_160/gammabatch_normalization_160/beta#batch_normalization_160/moving_mean'batch_normalization_160/moving_varianceconv2d_170/kernelconv2d_170/biasbatch_normalization_161/gammabatch_normalization_161/beta#batch_normalization_161/moving_mean'batch_normalization_161/moving_varianceconv2d_171/kernelconv2d_171/biasbatch_normalization_162/gammabatch_normalization_162/beta#batch_normalization_162/moving_mean'batch_normalization_162/moving_variancedense_37/kerneldense_37/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_423542
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Â"
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_167/kernel/Read/ReadVariableOp#conv2d_167/bias/Read/ReadVariableOp1batch_normalization_158/gamma/Read/ReadVariableOp0batch_normalization_158/beta/Read/ReadVariableOp7batch_normalization_158/moving_mean/Read/ReadVariableOp;batch_normalization_158/moving_variance/Read/ReadVariableOp%conv2d_168/kernel/Read/ReadVariableOp#conv2d_168/bias/Read/ReadVariableOp1batch_normalization_159/gamma/Read/ReadVariableOp0batch_normalization_159/beta/Read/ReadVariableOp7batch_normalization_159/moving_mean/Read/ReadVariableOp;batch_normalization_159/moving_variance/Read/ReadVariableOp%conv2d_169/kernel/Read/ReadVariableOp#conv2d_169/bias/Read/ReadVariableOp1batch_normalization_160/gamma/Read/ReadVariableOp0batch_normalization_160/beta/Read/ReadVariableOp7batch_normalization_160/moving_mean/Read/ReadVariableOp;batch_normalization_160/moving_variance/Read/ReadVariableOp%conv2d_170/kernel/Read/ReadVariableOp#conv2d_170/bias/Read/ReadVariableOp1batch_normalization_161/gamma/Read/ReadVariableOp0batch_normalization_161/beta/Read/ReadVariableOp7batch_normalization_161/moving_mean/Read/ReadVariableOp;batch_normalization_161/moving_variance/Read/ReadVariableOp%conv2d_171/kernel/Read/ReadVariableOp#conv2d_171/bias/Read/ReadVariableOp1batch_normalization_162/gamma/Read/ReadVariableOp0batch_normalization_162/beta/Read/ReadVariableOp7batch_normalization_162/moving_mean/Read/ReadVariableOp;batch_normalization_162/moving_variance/Read/ReadVariableOp#dense_37/kernel/Read/ReadVariableOp!dense_37/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/conv2d_167/kernel/m/Read/ReadVariableOp*Adam/conv2d_167/bias/m/Read/ReadVariableOp8Adam/batch_normalization_158/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_158/beta/m/Read/ReadVariableOp,Adam/conv2d_168/kernel/m/Read/ReadVariableOp*Adam/conv2d_168/bias/m/Read/ReadVariableOp8Adam/batch_normalization_159/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_159/beta/m/Read/ReadVariableOp,Adam/conv2d_169/kernel/m/Read/ReadVariableOp*Adam/conv2d_169/bias/m/Read/ReadVariableOp8Adam/batch_normalization_160/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_160/beta/m/Read/ReadVariableOp,Adam/conv2d_170/kernel/m/Read/ReadVariableOp*Adam/conv2d_170/bias/m/Read/ReadVariableOp8Adam/batch_normalization_161/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_161/beta/m/Read/ReadVariableOp,Adam/conv2d_171/kernel/m/Read/ReadVariableOp*Adam/conv2d_171/bias/m/Read/ReadVariableOp8Adam/batch_normalization_162/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_162/beta/m/Read/ReadVariableOp*Adam/dense_37/kernel/m/Read/ReadVariableOp(Adam/dense_37/bias/m/Read/ReadVariableOp,Adam/conv2d_167/kernel/v/Read/ReadVariableOp*Adam/conv2d_167/bias/v/Read/ReadVariableOp8Adam/batch_normalization_158/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_158/beta/v/Read/ReadVariableOp,Adam/conv2d_168/kernel/v/Read/ReadVariableOp*Adam/conv2d_168/bias/v/Read/ReadVariableOp8Adam/batch_normalization_159/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_159/beta/v/Read/ReadVariableOp,Adam/conv2d_169/kernel/v/Read/ReadVariableOp*Adam/conv2d_169/bias/v/Read/ReadVariableOp8Adam/batch_normalization_160/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_160/beta/v/Read/ReadVariableOp,Adam/conv2d_170/kernel/v/Read/ReadVariableOp*Adam/conv2d_170/bias/v/Read/ReadVariableOp8Adam/batch_normalization_161/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_161/beta/v/Read/ReadVariableOp,Adam/conv2d_171/kernel/v/Read/ReadVariableOp*Adam/conv2d_171/bias/v/Read/ReadVariableOp8Adam/batch_normalization_162/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_162/beta/v/Read/ReadVariableOp*Adam/dense_37/kernel/v/Read/ReadVariableOp(Adam/dense_37/bias/v/Read/ReadVariableOpConst*b
Tin[
Y2W	*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_424338

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_167/kernelconv2d_167/biasbatch_normalization_158/gammabatch_normalization_158/beta#batch_normalization_158/moving_mean'batch_normalization_158/moving_varianceconv2d_168/kernelconv2d_168/biasbatch_normalization_159/gammabatch_normalization_159/beta#batch_normalization_159/moving_mean'batch_normalization_159/moving_varianceconv2d_169/kernelconv2d_169/biasbatch_normalization_160/gammabatch_normalization_160/beta#batch_normalization_160/moving_mean'batch_normalization_160/moving_varianceconv2d_170/kernelconv2d_170/biasbatch_normalization_161/gammabatch_normalization_161/beta#batch_normalization_161/moving_mean'batch_normalization_161/moving_varianceconv2d_171/kernelconv2d_171/biasbatch_normalization_162/gammabatch_normalization_162/beta#batch_normalization_162/moving_mean'batch_normalization_162/moving_variancedense_37/kerneldense_37/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_167/kernel/mAdam/conv2d_167/bias/m$Adam/batch_normalization_158/gamma/m#Adam/batch_normalization_158/beta/mAdam/conv2d_168/kernel/mAdam/conv2d_168/bias/m$Adam/batch_normalization_159/gamma/m#Adam/batch_normalization_159/beta/mAdam/conv2d_169/kernel/mAdam/conv2d_169/bias/m$Adam/batch_normalization_160/gamma/m#Adam/batch_normalization_160/beta/mAdam/conv2d_170/kernel/mAdam/conv2d_170/bias/m$Adam/batch_normalization_161/gamma/m#Adam/batch_normalization_161/beta/mAdam/conv2d_171/kernel/mAdam/conv2d_171/bias/m$Adam/batch_normalization_162/gamma/m#Adam/batch_normalization_162/beta/mAdam/dense_37/kernel/mAdam/dense_37/bias/mAdam/conv2d_167/kernel/vAdam/conv2d_167/bias/v$Adam/batch_normalization_158/gamma/v#Adam/batch_normalization_158/beta/vAdam/conv2d_168/kernel/vAdam/conv2d_168/bias/v$Adam/batch_normalization_159/gamma/v#Adam/batch_normalization_159/beta/vAdam/conv2d_169/kernel/vAdam/conv2d_169/bias/v$Adam/batch_normalization_160/gamma/v#Adam/batch_normalization_160/beta/vAdam/conv2d_170/kernel/vAdam/conv2d_170/bias/v$Adam/batch_normalization_161/gamma/v#Adam/batch_normalization_161/beta/vAdam/conv2d_171/kernel/vAdam/conv2d_171/bias/v$Adam/batch_normalization_162/gamma/v#Adam/batch_normalization_162/beta/vAdam/dense_37/kernel/vAdam/dense_37/bias/v*a
TinZ
X2V*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_424603Ôí
¼
N
2__inference_max_pooling2d_171_layer_call_fn_423997

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_171_layer_call_and_return_conditional_losses_422273
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î

S__inference_batch_normalization_161_layer_call_and_return_conditional_losses_422146

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ö
 
+__inference_conv2d_168_layer_call_fn_423643

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_168_layer_call_and_return_conditional_losses_422321y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ×: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
 
_user_specified_nameinputs
	
Ó
8__inference_batch_normalization_158_layer_call_fn_423575

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_158_layer_call_and_return_conditional_losses_421918
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
]
Ô
I__inference_sequential_21_layer_call_and_return_conditional_losses_423072
conv2d_167_input+
conv2d_167_422989:
conv2d_167_422991:,
batch_normalization_158_422994:,
batch_normalization_158_422996:,
batch_normalization_158_422998:,
batch_normalization_158_423000:+
conv2d_168_423004:
conv2d_168_423006:,
batch_normalization_159_423009:,
batch_normalization_159_423011:,
batch_normalization_159_423013:,
batch_normalization_159_423015:+
conv2d_169_423019: 
conv2d_169_423021: ,
batch_normalization_160_423024: ,
batch_normalization_160_423026: ,
batch_normalization_160_423028: ,
batch_normalization_160_423030: +
conv2d_170_423034: @
conv2d_170_423036:@,
batch_normalization_161_423039:@,
batch_normalization_161_423041:@,
batch_normalization_161_423043:@,
batch_normalization_161_423045:@,
conv2d_171_423049:@ 
conv2d_171_423051:	-
batch_normalization_162_423054:	-
batch_normalization_162_423056:	-
batch_normalization_162_423058:	-
batch_normalization_162_423060:	"
dense_37_423066:	M

dense_37_423068:

identity¢/batch_normalization_158/StatefulPartitionedCall¢/batch_normalization_159/StatefulPartitionedCall¢/batch_normalization_160/StatefulPartitionedCall¢/batch_normalization_161/StatefulPartitionedCall¢/batch_normalization_162/StatefulPartitionedCall¢"conv2d_167/StatefulPartitionedCall¢"conv2d_168/StatefulPartitionedCall¢"conv2d_169/StatefulPartitionedCall¢"conv2d_170/StatefulPartitionedCall¢"conv2d_171/StatefulPartitionedCall¢ dense_37/StatefulPartitionedCall¢"dropout_40/StatefulPartitionedCall
"conv2d_167/StatefulPartitionedCallStatefulPartitionedCallconv2d_167_inputconv2d_167_422989conv2d_167_422991*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_167_layer_call_and_return_conditional_losses_422294
/batch_normalization_158/StatefulPartitionedCallStatefulPartitionedCall+conv2d_167/StatefulPartitionedCall:output:0batch_normalization_158_422994batch_normalization_158_422996batch_normalization_158_422998batch_normalization_158_423000*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_158_layer_call_and_return_conditional_losses_421949
!max_pooling2d_167/PartitionedCallPartitionedCall8batch_normalization_158/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_167_layer_call_and_return_conditional_losses_421969¦
"conv2d_168/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_167/PartitionedCall:output:0conv2d_168_423004conv2d_168_423006*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_168_layer_call_and_return_conditional_losses_422321
/batch_normalization_159/StatefulPartitionedCallStatefulPartitionedCall+conv2d_168/StatefulPartitionedCall:output:0batch_normalization_159_423009batch_normalization_159_423011batch_normalization_159_423013batch_normalization_159_423015*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_159_layer_call_and_return_conditional_losses_422025
!max_pooling2d_168/PartitionedCallPartitionedCall8batch_normalization_159/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿFj* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_168_layer_call_and_return_conditional_losses_422045¤
"conv2d_169/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_168/PartitionedCall:output:0conv2d_169_423019conv2d_169_423021*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_169_layer_call_and_return_conditional_losses_422348
/batch_normalization_160/StatefulPartitionedCallStatefulPartitionedCall+conv2d_169/StatefulPartitionedCall:output:0batch_normalization_160_423024batch_normalization_160_423026batch_normalization_160_423028batch_normalization_160_423030*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_160_layer_call_and_return_conditional_losses_422101
!max_pooling2d_169/PartitionedCallPartitionedCall8batch_normalization_160/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"4 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_169_layer_call_and_return_conditional_losses_422121¤
"conv2d_170/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_169/PartitionedCall:output:0conv2d_170_423034conv2d_170_423036*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_170_layer_call_and_return_conditional_losses_422375
/batch_normalization_161/StatefulPartitionedCallStatefulPartitionedCall+conv2d_170/StatefulPartitionedCall:output:0batch_normalization_161_423039batch_normalization_161_423041batch_normalization_161_423043batch_normalization_161_423045*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_161_layer_call_and_return_conditional_losses_422177
!max_pooling2d_170/PartitionedCallPartitionedCall8batch_normalization_161/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_170_layer_call_and_return_conditional_losses_422197¥
"conv2d_171/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_170/PartitionedCall:output:0conv2d_171_423049conv2d_171_423051*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_171_layer_call_and_return_conditional_losses_422402
/batch_normalization_162/StatefulPartitionedCallStatefulPartitionedCall+conv2d_171/StatefulPartitionedCall:output:0batch_normalization_162_423054batch_normalization_162_423056batch_normalization_162_423058batch_normalization_162_423060*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_162_layer_call_and_return_conditional_losses_422253
!max_pooling2d_171/PartitionedCallPartitionedCall8batch_normalization_162/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_171_layer_call_and_return_conditional_losses_422273á
flatten_38/PartitionedCallPartitionedCall*max_pooling2d_171/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_38_layer_call_and_return_conditional_losses_422424ê
"dropout_40/StatefulPartitionedCallStatefulPartitionedCall#flatten_38/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_40_layer_call_and_return_conditional_losses_422548
 dense_37/StatefulPartitionedCallStatefulPartitionedCall+dropout_40/StatefulPartitionedCall:output:0dense_37_423066dense_37_423068*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_422444x
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Á
NoOpNoOp0^batch_normalization_158/StatefulPartitionedCall0^batch_normalization_159/StatefulPartitionedCall0^batch_normalization_160/StatefulPartitionedCall0^batch_normalization_161/StatefulPartitionedCall0^batch_normalization_162/StatefulPartitionedCall#^conv2d_167/StatefulPartitionedCall#^conv2d_168/StatefulPartitionedCall#^conv2d_169/StatefulPartitionedCall#^conv2d_170/StatefulPartitionedCall#^conv2d_171/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall#^dropout_40/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ °: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_158/StatefulPartitionedCall/batch_normalization_158/StatefulPartitionedCall2b
/batch_normalization_159/StatefulPartitionedCall/batch_normalization_159/StatefulPartitionedCall2b
/batch_normalization_160/StatefulPartitionedCall/batch_normalization_160/StatefulPartitionedCall2b
/batch_normalization_161/StatefulPartitionedCall/batch_normalization_161/StatefulPartitionedCall2b
/batch_normalization_162/StatefulPartitionedCall/batch_normalization_162/StatefulPartitionedCall2H
"conv2d_167/StatefulPartitionedCall"conv2d_167/StatefulPartitionedCall2H
"conv2d_168/StatefulPartitionedCall"conv2d_168/StatefulPartitionedCall2H
"conv2d_169/StatefulPartitionedCall"conv2d_169/StatefulPartitionedCall2H
"conv2d_170/StatefulPartitionedCall"conv2d_170/StatefulPartitionedCall2H
"conv2d_171/StatefulPartitionedCall"conv2d_171/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2H
"dropout_40/StatefulPartitionedCall"dropout_40/StatefulPartitionedCall:c _
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
*
_user_specified_nameconv2d_167_input
¼
N
2__inference_max_pooling2d_167_layer_call_fn_423629

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_167_layer_call_and_return_conditional_losses_421969
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á[
¯
I__inference_sequential_21_layer_call_and_return_conditional_losses_422986
conv2d_167_input+
conv2d_167_422903:
conv2d_167_422905:,
batch_normalization_158_422908:,
batch_normalization_158_422910:,
batch_normalization_158_422912:,
batch_normalization_158_422914:+
conv2d_168_422918:
conv2d_168_422920:,
batch_normalization_159_422923:,
batch_normalization_159_422925:,
batch_normalization_159_422927:,
batch_normalization_159_422929:+
conv2d_169_422933: 
conv2d_169_422935: ,
batch_normalization_160_422938: ,
batch_normalization_160_422940: ,
batch_normalization_160_422942: ,
batch_normalization_160_422944: +
conv2d_170_422948: @
conv2d_170_422950:@,
batch_normalization_161_422953:@,
batch_normalization_161_422955:@,
batch_normalization_161_422957:@,
batch_normalization_161_422959:@,
conv2d_171_422963:@ 
conv2d_171_422965:	-
batch_normalization_162_422968:	-
batch_normalization_162_422970:	-
batch_normalization_162_422972:	-
batch_normalization_162_422974:	"
dense_37_422980:	M

dense_37_422982:

identity¢/batch_normalization_158/StatefulPartitionedCall¢/batch_normalization_159/StatefulPartitionedCall¢/batch_normalization_160/StatefulPartitionedCall¢/batch_normalization_161/StatefulPartitionedCall¢/batch_normalization_162/StatefulPartitionedCall¢"conv2d_167/StatefulPartitionedCall¢"conv2d_168/StatefulPartitionedCall¢"conv2d_169/StatefulPartitionedCall¢"conv2d_170/StatefulPartitionedCall¢"conv2d_171/StatefulPartitionedCall¢ dense_37/StatefulPartitionedCall
"conv2d_167/StatefulPartitionedCallStatefulPartitionedCallconv2d_167_inputconv2d_167_422903conv2d_167_422905*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_167_layer_call_and_return_conditional_losses_422294
/batch_normalization_158/StatefulPartitionedCallStatefulPartitionedCall+conv2d_167/StatefulPartitionedCall:output:0batch_normalization_158_422908batch_normalization_158_422910batch_normalization_158_422912batch_normalization_158_422914*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_158_layer_call_and_return_conditional_losses_421918
!max_pooling2d_167/PartitionedCallPartitionedCall8batch_normalization_158/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_167_layer_call_and_return_conditional_losses_421969¦
"conv2d_168/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_167/PartitionedCall:output:0conv2d_168_422918conv2d_168_422920*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_168_layer_call_and_return_conditional_losses_422321
/batch_normalization_159/StatefulPartitionedCallStatefulPartitionedCall+conv2d_168/StatefulPartitionedCall:output:0batch_normalization_159_422923batch_normalization_159_422925batch_normalization_159_422927batch_normalization_159_422929*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_159_layer_call_and_return_conditional_losses_421994
!max_pooling2d_168/PartitionedCallPartitionedCall8batch_normalization_159/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿFj* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_168_layer_call_and_return_conditional_losses_422045¤
"conv2d_169/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_168/PartitionedCall:output:0conv2d_169_422933conv2d_169_422935*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_169_layer_call_and_return_conditional_losses_422348
/batch_normalization_160/StatefulPartitionedCallStatefulPartitionedCall+conv2d_169/StatefulPartitionedCall:output:0batch_normalization_160_422938batch_normalization_160_422940batch_normalization_160_422942batch_normalization_160_422944*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_160_layer_call_and_return_conditional_losses_422070
!max_pooling2d_169/PartitionedCallPartitionedCall8batch_normalization_160/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"4 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_169_layer_call_and_return_conditional_losses_422121¤
"conv2d_170/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_169/PartitionedCall:output:0conv2d_170_422948conv2d_170_422950*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_170_layer_call_and_return_conditional_losses_422375
/batch_normalization_161/StatefulPartitionedCallStatefulPartitionedCall+conv2d_170/StatefulPartitionedCall:output:0batch_normalization_161_422953batch_normalization_161_422955batch_normalization_161_422957batch_normalization_161_422959*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_161_layer_call_and_return_conditional_losses_422146
!max_pooling2d_170/PartitionedCallPartitionedCall8batch_normalization_161/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_170_layer_call_and_return_conditional_losses_422197¥
"conv2d_171/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_170/PartitionedCall:output:0conv2d_171_422963conv2d_171_422965*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_171_layer_call_and_return_conditional_losses_422402
/batch_normalization_162/StatefulPartitionedCallStatefulPartitionedCall+conv2d_171/StatefulPartitionedCall:output:0batch_normalization_162_422968batch_normalization_162_422970batch_normalization_162_422972batch_normalization_162_422974*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_162_layer_call_and_return_conditional_losses_422222
!max_pooling2d_171/PartitionedCallPartitionedCall8batch_normalization_162/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_171_layer_call_and_return_conditional_losses_422273á
flatten_38/PartitionedCallPartitionedCall*max_pooling2d_171/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_38_layer_call_and_return_conditional_losses_422424Ú
dropout_40/PartitionedCallPartitionedCall#flatten_38/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_40_layer_call_and_return_conditional_losses_422431
 dense_37/StatefulPartitionedCallStatefulPartitionedCall#dropout_40/PartitionedCall:output:0dense_37_422980dense_37_422982*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_422444x
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp0^batch_normalization_158/StatefulPartitionedCall0^batch_normalization_159/StatefulPartitionedCall0^batch_normalization_160/StatefulPartitionedCall0^batch_normalization_161/StatefulPartitionedCall0^batch_normalization_162/StatefulPartitionedCall#^conv2d_167/StatefulPartitionedCall#^conv2d_168/StatefulPartitionedCall#^conv2d_169/StatefulPartitionedCall#^conv2d_170/StatefulPartitionedCall#^conv2d_171/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ °: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_158/StatefulPartitionedCall/batch_normalization_158/StatefulPartitionedCall2b
/batch_normalization_159/StatefulPartitionedCall/batch_normalization_159/StatefulPartitionedCall2b
/batch_normalization_160/StatefulPartitionedCall/batch_normalization_160/StatefulPartitionedCall2b
/batch_normalization_161/StatefulPartitionedCall/batch_normalization_161/StatefulPartitionedCall2b
/batch_normalization_162/StatefulPartitionedCall/batch_normalization_162/StatefulPartitionedCall2H
"conv2d_167/StatefulPartitionedCall"conv2d_167/StatefulPartitionedCall2H
"conv2d_168/StatefulPartitionedCall"conv2d_168/StatefulPartitionedCall2H
"conv2d_169/StatefulPartitionedCall"conv2d_169/StatefulPartitionedCall2H
"conv2d_170/StatefulPartitionedCall"conv2d_170/StatefulPartitionedCall2H
"conv2d_171/StatefulPartitionedCall"conv2d_171/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:c _
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
*
_user_specified_nameconv2d_167_input
ö
 
+__inference_conv2d_167_layer_call_fn_423551

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_167_layer_call_and_return_conditional_losses_422294y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ °: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_168_layer_call_and_return_conditional_losses_422321

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕk
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ×: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
 
_user_specified_nameinputs
ü	
e
F__inference_dropout_40_layer_call_and_return_conditional_losses_424040

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿMC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿMp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿMj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿMZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿM:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
 
_user_specified_nameinputs
°Ç
Ã 
I__inference_sequential_21_layer_call_and_return_conditional_losses_423471

inputsC
)conv2d_167_conv2d_readvariableop_resource:8
*conv2d_167_biasadd_readvariableop_resource:=
/batch_normalization_158_readvariableop_resource:?
1batch_normalization_158_readvariableop_1_resource:N
@batch_normalization_158_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_158_fusedbatchnormv3_readvariableop_1_resource:C
)conv2d_168_conv2d_readvariableop_resource:8
*conv2d_168_biasadd_readvariableop_resource:=
/batch_normalization_159_readvariableop_resource:?
1batch_normalization_159_readvariableop_1_resource:N
@batch_normalization_159_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_159_fusedbatchnormv3_readvariableop_1_resource:C
)conv2d_169_conv2d_readvariableop_resource: 8
*conv2d_169_biasadd_readvariableop_resource: =
/batch_normalization_160_readvariableop_resource: ?
1batch_normalization_160_readvariableop_1_resource: N
@batch_normalization_160_fusedbatchnormv3_readvariableop_resource: P
Bbatch_normalization_160_fusedbatchnormv3_readvariableop_1_resource: C
)conv2d_170_conv2d_readvariableop_resource: @8
*conv2d_170_biasadd_readvariableop_resource:@=
/batch_normalization_161_readvariableop_resource:@?
1batch_normalization_161_readvariableop_1_resource:@N
@batch_normalization_161_fusedbatchnormv3_readvariableop_resource:@P
Bbatch_normalization_161_fusedbatchnormv3_readvariableop_1_resource:@D
)conv2d_171_conv2d_readvariableop_resource:@9
*conv2d_171_biasadd_readvariableop_resource:	>
/batch_normalization_162_readvariableop_resource:	@
1batch_normalization_162_readvariableop_1_resource:	O
@batch_normalization_162_fusedbatchnormv3_readvariableop_resource:	Q
Bbatch_normalization_162_fusedbatchnormv3_readvariableop_1_resource:	:
'dense_37_matmul_readvariableop_resource:	M
6
(dense_37_biasadd_readvariableop_resource:

identity¢&batch_normalization_158/AssignNewValue¢(batch_normalization_158/AssignNewValue_1¢7batch_normalization_158/FusedBatchNormV3/ReadVariableOp¢9batch_normalization_158/FusedBatchNormV3/ReadVariableOp_1¢&batch_normalization_158/ReadVariableOp¢(batch_normalization_158/ReadVariableOp_1¢&batch_normalization_159/AssignNewValue¢(batch_normalization_159/AssignNewValue_1¢7batch_normalization_159/FusedBatchNormV3/ReadVariableOp¢9batch_normalization_159/FusedBatchNormV3/ReadVariableOp_1¢&batch_normalization_159/ReadVariableOp¢(batch_normalization_159/ReadVariableOp_1¢&batch_normalization_160/AssignNewValue¢(batch_normalization_160/AssignNewValue_1¢7batch_normalization_160/FusedBatchNormV3/ReadVariableOp¢9batch_normalization_160/FusedBatchNormV3/ReadVariableOp_1¢&batch_normalization_160/ReadVariableOp¢(batch_normalization_160/ReadVariableOp_1¢&batch_normalization_161/AssignNewValue¢(batch_normalization_161/AssignNewValue_1¢7batch_normalization_161/FusedBatchNormV3/ReadVariableOp¢9batch_normalization_161/FusedBatchNormV3/ReadVariableOp_1¢&batch_normalization_161/ReadVariableOp¢(batch_normalization_161/ReadVariableOp_1¢&batch_normalization_162/AssignNewValue¢(batch_normalization_162/AssignNewValue_1¢7batch_normalization_162/FusedBatchNormV3/ReadVariableOp¢9batch_normalization_162/FusedBatchNormV3/ReadVariableOp_1¢&batch_normalization_162/ReadVariableOp¢(batch_normalization_162/ReadVariableOp_1¢!conv2d_167/BiasAdd/ReadVariableOp¢ conv2d_167/Conv2D/ReadVariableOp¢!conv2d_168/BiasAdd/ReadVariableOp¢ conv2d_168/Conv2D/ReadVariableOp¢!conv2d_169/BiasAdd/ReadVariableOp¢ conv2d_169/Conv2D/ReadVariableOp¢!conv2d_170/BiasAdd/ReadVariableOp¢ conv2d_170/Conv2D/ReadVariableOp¢!conv2d_171/BiasAdd/ReadVariableOp¢ conv2d_171/Conv2D/ReadVariableOp¢dense_37/BiasAdd/ReadVariableOp¢dense_37/MatMul/ReadVariableOp
 conv2d_167/Conv2D/ReadVariableOpReadVariableOp)conv2d_167_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0²
conv2d_167/Conv2DConv2Dinputs(conv2d_167/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®*
paddingVALID*
strides

!conv2d_167/BiasAdd/ReadVariableOpReadVariableOp*conv2d_167_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
conv2d_167/BiasAddBiasAddconv2d_167/Conv2D:output:0)conv2d_167/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®p
conv2d_167/ReluReluconv2d_167/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
&batch_normalization_158/ReadVariableOpReadVariableOp/batch_normalization_158_readvariableop_resource*
_output_shapes
:*
dtype0
(batch_normalization_158/ReadVariableOp_1ReadVariableOp1batch_normalization_158_readvariableop_1_resource*
_output_shapes
:*
dtype0´
7batch_normalization_158/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_158_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¸
9batch_normalization_158/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_158_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Õ
(batch_normalization_158/FusedBatchNormV3FusedBatchNormV3conv2d_167/Relu:activations:0.batch_normalization_158/ReadVariableOp:value:00batch_normalization_158/ReadVariableOp_1:value:0?batch_normalization_158/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_158/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ®:::::*
epsilon%o:*
exponential_avg_factor%
×#<
&batch_normalization_158/AssignNewValueAssignVariableOp@batch_normalization_158_fusedbatchnormv3_readvariableop_resource5batch_normalization_158/FusedBatchNormV3:batch_mean:08^batch_normalization_158/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
(batch_normalization_158/AssignNewValue_1AssignVariableOpBbatch_normalization_158_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_158/FusedBatchNormV3:batch_variance:0:^batch_normalization_158/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0Á
max_pooling2d_167/MaxPoolMaxPool,batch_normalization_158/FusedBatchNormV3:y:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*
ksize
*
paddingVALID*
strides

 conv2d_168/Conv2D/ReadVariableOpReadVariableOp)conv2d_168_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Î
conv2d_168/Conv2DConv2D"max_pooling2d_167/MaxPool:output:0(conv2d_168/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ*
paddingVALID*
strides

!conv2d_168/BiasAdd/ReadVariableOpReadVariableOp*conv2d_168_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
conv2d_168/BiasAddBiasAddconv2d_168/Conv2D:output:0)conv2d_168/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕp
conv2d_168/ReluReluconv2d_168/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
&batch_normalization_159/ReadVariableOpReadVariableOp/batch_normalization_159_readvariableop_resource*
_output_shapes
:*
dtype0
(batch_normalization_159/ReadVariableOp_1ReadVariableOp1batch_normalization_159_readvariableop_1_resource*
_output_shapes
:*
dtype0´
7batch_normalization_159/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_159_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¸
9batch_normalization_159/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_159_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Õ
(batch_normalization_159/FusedBatchNormV3FusedBatchNormV3conv2d_168/Relu:activations:0.batch_normalization_159/ReadVariableOp:value:00batch_normalization_159/ReadVariableOp_1:value:0?batch_normalization_159/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_159/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿÕ:::::*
epsilon%o:*
exponential_avg_factor%
×#<
&batch_normalization_159/AssignNewValueAssignVariableOp@batch_normalization_159_fusedbatchnormv3_readvariableop_resource5batch_normalization_159/FusedBatchNormV3:batch_mean:08^batch_normalization_159/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
(batch_normalization_159/AssignNewValue_1AssignVariableOpBbatch_normalization_159_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_159/FusedBatchNormV3:batch_variance:0:^batch_normalization_159/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0¿
max_pooling2d_168/MaxPoolMaxPool,batch_normalization_159/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFj*
ksize
*
paddingVALID*
strides

 conv2d_169/Conv2D/ReadVariableOpReadVariableOp)conv2d_169_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ì
conv2d_169/Conv2DConv2D"max_pooling2d_168/MaxPool:output:0(conv2d_169/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh *
paddingVALID*
strides

!conv2d_169/BiasAdd/ReadVariableOpReadVariableOp*conv2d_169_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_169/BiasAddBiasAddconv2d_169/Conv2D:output:0)conv2d_169/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh n
conv2d_169/ReluReluconv2d_169/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh 
&batch_normalization_160/ReadVariableOpReadVariableOp/batch_normalization_160_readvariableop_resource*
_output_shapes
: *
dtype0
(batch_normalization_160/ReadVariableOp_1ReadVariableOp1batch_normalization_160_readvariableop_1_resource*
_output_shapes
: *
dtype0´
7batch_normalization_160/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_160_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¸
9batch_normalization_160/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_160_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ó
(batch_normalization_160/FusedBatchNormV3FusedBatchNormV3conv2d_169/Relu:activations:0.batch_normalization_160/ReadVariableOp:value:00batch_normalization_160/ReadVariableOp_1:value:0?batch_normalization_160/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_160/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿDh : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<
&batch_normalization_160/AssignNewValueAssignVariableOp@batch_normalization_160_fusedbatchnormv3_readvariableop_resource5batch_normalization_160/FusedBatchNormV3:batch_mean:08^batch_normalization_160/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
(batch_normalization_160/AssignNewValue_1AssignVariableOpBbatch_normalization_160_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_160/FusedBatchNormV3:batch_variance:0:^batch_normalization_160/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0¿
max_pooling2d_169/MaxPoolMaxPool,batch_normalization_160/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"4 *
ksize
*
paddingVALID*
strides

 conv2d_170/Conv2D/ReadVariableOpReadVariableOp)conv2d_170_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ì
conv2d_170/Conv2DConv2D"max_pooling2d_169/MaxPool:output:0(conv2d_170/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@*
paddingVALID*
strides

!conv2d_170/BiasAdd/ReadVariableOpReadVariableOp*conv2d_170_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_170/BiasAddBiasAddconv2d_170/Conv2D:output:0)conv2d_170/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@n
conv2d_170/ReluReluconv2d_170/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@
&batch_normalization_161/ReadVariableOpReadVariableOp/batch_normalization_161_readvariableop_resource*
_output_shapes
:@*
dtype0
(batch_normalization_161/ReadVariableOp_1ReadVariableOp1batch_normalization_161_readvariableop_1_resource*
_output_shapes
:@*
dtype0´
7batch_normalization_161/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_161_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¸
9batch_normalization_161/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_161_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ó
(batch_normalization_161/FusedBatchNormV3FusedBatchNormV3conv2d_170/Relu:activations:0.batch_normalization_161/ReadVariableOp:value:00batch_normalization_161/ReadVariableOp_1:value:0?batch_normalization_161/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_161/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ 2@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<
&batch_normalization_161/AssignNewValueAssignVariableOp@batch_normalization_161_fusedbatchnormv3_readvariableop_resource5batch_normalization_161/FusedBatchNormV3:batch_mean:08^batch_normalization_161/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
(batch_normalization_161/AssignNewValue_1AssignVariableOpBbatch_normalization_161_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_161/FusedBatchNormV3:batch_variance:0:^batch_normalization_161/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0¿
max_pooling2d_170/MaxPoolMaxPool,batch_normalization_161/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

 conv2d_171/Conv2D/ReadVariableOpReadVariableOp)conv2d_171_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Í
conv2d_171/Conv2DConv2D"max_pooling2d_170/MaxPool:output:0(conv2d_171/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

!conv2d_171/BiasAdd/ReadVariableOpReadVariableOp*conv2d_171_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_171/BiasAddBiasAddconv2d_171/Conv2D:output:0)conv2d_171/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
conv2d_171/ReluReluconv2d_171/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&batch_normalization_162/ReadVariableOpReadVariableOp/batch_normalization_162_readvariableop_resource*
_output_shapes	
:*
dtype0
(batch_normalization_162/ReadVariableOp_1ReadVariableOp1batch_normalization_162_readvariableop_1_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_162/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_162_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0¹
9batch_normalization_162/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_162_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ø
(batch_normalization_162/FusedBatchNormV3FusedBatchNormV3conv2d_171/Relu:activations:0.batch_normalization_162/ReadVariableOp:value:00batch_normalization_162/ReadVariableOp_1:value:0?batch_normalization_162/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_162/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<
&batch_normalization_162/AssignNewValueAssignVariableOp@batch_normalization_162_fusedbatchnormv3_readvariableop_resource5batch_normalization_162/FusedBatchNormV3:batch_mean:08^batch_normalization_162/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
(batch_normalization_162/AssignNewValue_1AssignVariableOpBbatch_normalization_162_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_162/FusedBatchNormV3:batch_variance:0:^batch_normalization_162/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0À
max_pooling2d_171/MaxPoolMaxPool,batch_normalization_162/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
a
flatten_38/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&  
flatten_38/ReshapeReshape"max_pooling2d_171/MaxPool:output:0flatten_38/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM]
dropout_40/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_40/dropout/MulMulflatten_38/Reshape:output:0!dropout_40/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿMc
dropout_40/dropout/ShapeShapeflatten_38/Reshape:output:0*
T0*
_output_shapes
:£
/dropout_40/dropout/random_uniform/RandomUniformRandomUniform!dropout_40/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM*
dtype0f
!dropout_40/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?È
dropout_40/dropout/GreaterEqualGreaterEqual8dropout_40/dropout/random_uniform/RandomUniform:output:0*dropout_40/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
dropout_40/dropout/CastCast#dropout_40/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
dropout_40/dropout/Mul_1Muldropout_40/dropout/Mul:z:0dropout_40/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes
:	M
*
dtype0
dense_37/MatMulMatMuldropout_40/dropout/Mul_1:z:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
dense_37/SoftmaxSoftmaxdense_37/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
IdentityIdentitydense_37/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp'^batch_normalization_158/AssignNewValue)^batch_normalization_158/AssignNewValue_18^batch_normalization_158/FusedBatchNormV3/ReadVariableOp:^batch_normalization_158/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_158/ReadVariableOp)^batch_normalization_158/ReadVariableOp_1'^batch_normalization_159/AssignNewValue)^batch_normalization_159/AssignNewValue_18^batch_normalization_159/FusedBatchNormV3/ReadVariableOp:^batch_normalization_159/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_159/ReadVariableOp)^batch_normalization_159/ReadVariableOp_1'^batch_normalization_160/AssignNewValue)^batch_normalization_160/AssignNewValue_18^batch_normalization_160/FusedBatchNormV3/ReadVariableOp:^batch_normalization_160/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_160/ReadVariableOp)^batch_normalization_160/ReadVariableOp_1'^batch_normalization_161/AssignNewValue)^batch_normalization_161/AssignNewValue_18^batch_normalization_161/FusedBatchNormV3/ReadVariableOp:^batch_normalization_161/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_161/ReadVariableOp)^batch_normalization_161/ReadVariableOp_1'^batch_normalization_162/AssignNewValue)^batch_normalization_162/AssignNewValue_18^batch_normalization_162/FusedBatchNormV3/ReadVariableOp:^batch_normalization_162/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_162/ReadVariableOp)^batch_normalization_162/ReadVariableOp_1"^conv2d_167/BiasAdd/ReadVariableOp!^conv2d_167/Conv2D/ReadVariableOp"^conv2d_168/BiasAdd/ReadVariableOp!^conv2d_168/Conv2D/ReadVariableOp"^conv2d_169/BiasAdd/ReadVariableOp!^conv2d_169/Conv2D/ReadVariableOp"^conv2d_170/BiasAdd/ReadVariableOp!^conv2d_170/Conv2D/ReadVariableOp"^conv2d_171/BiasAdd/ReadVariableOp!^conv2d_171/Conv2D/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp^dense_37/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ °: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_158/AssignNewValue&batch_normalization_158/AssignNewValue2T
(batch_normalization_158/AssignNewValue_1(batch_normalization_158/AssignNewValue_12r
7batch_normalization_158/FusedBatchNormV3/ReadVariableOp7batch_normalization_158/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_158/FusedBatchNormV3/ReadVariableOp_19batch_normalization_158/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_158/ReadVariableOp&batch_normalization_158/ReadVariableOp2T
(batch_normalization_158/ReadVariableOp_1(batch_normalization_158/ReadVariableOp_12P
&batch_normalization_159/AssignNewValue&batch_normalization_159/AssignNewValue2T
(batch_normalization_159/AssignNewValue_1(batch_normalization_159/AssignNewValue_12r
7batch_normalization_159/FusedBatchNormV3/ReadVariableOp7batch_normalization_159/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_159/FusedBatchNormV3/ReadVariableOp_19batch_normalization_159/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_159/ReadVariableOp&batch_normalization_159/ReadVariableOp2T
(batch_normalization_159/ReadVariableOp_1(batch_normalization_159/ReadVariableOp_12P
&batch_normalization_160/AssignNewValue&batch_normalization_160/AssignNewValue2T
(batch_normalization_160/AssignNewValue_1(batch_normalization_160/AssignNewValue_12r
7batch_normalization_160/FusedBatchNormV3/ReadVariableOp7batch_normalization_160/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_160/FusedBatchNormV3/ReadVariableOp_19batch_normalization_160/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_160/ReadVariableOp&batch_normalization_160/ReadVariableOp2T
(batch_normalization_160/ReadVariableOp_1(batch_normalization_160/ReadVariableOp_12P
&batch_normalization_161/AssignNewValue&batch_normalization_161/AssignNewValue2T
(batch_normalization_161/AssignNewValue_1(batch_normalization_161/AssignNewValue_12r
7batch_normalization_161/FusedBatchNormV3/ReadVariableOp7batch_normalization_161/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_161/FusedBatchNormV3/ReadVariableOp_19batch_normalization_161/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_161/ReadVariableOp&batch_normalization_161/ReadVariableOp2T
(batch_normalization_161/ReadVariableOp_1(batch_normalization_161/ReadVariableOp_12P
&batch_normalization_162/AssignNewValue&batch_normalization_162/AssignNewValue2T
(batch_normalization_162/AssignNewValue_1(batch_normalization_162/AssignNewValue_12r
7batch_normalization_162/FusedBatchNormV3/ReadVariableOp7batch_normalization_162/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_162/FusedBatchNormV3/ReadVariableOp_19batch_normalization_162/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_162/ReadVariableOp&batch_normalization_162/ReadVariableOp2T
(batch_normalization_162/ReadVariableOp_1(batch_normalization_162/ReadVariableOp_12F
!conv2d_167/BiasAdd/ReadVariableOp!conv2d_167/BiasAdd/ReadVariableOp2D
 conv2d_167/Conv2D/ReadVariableOp conv2d_167/Conv2D/ReadVariableOp2F
!conv2d_168/BiasAdd/ReadVariableOp!conv2d_168/BiasAdd/ReadVariableOp2D
 conv2d_168/Conv2D/ReadVariableOp conv2d_168/Conv2D/ReadVariableOp2F
!conv2d_169/BiasAdd/ReadVariableOp!conv2d_169/BiasAdd/ReadVariableOp2D
 conv2d_169/Conv2D/ReadVariableOp conv2d_169/Conv2D/ReadVariableOp2F
!conv2d_170/BiasAdd/ReadVariableOp!conv2d_170/BiasAdd/ReadVariableOp2D
 conv2d_170/Conv2D/ReadVariableOp conv2d_170/Conv2D/ReadVariableOp2F
!conv2d_171/BiasAdd/ReadVariableOp!conv2d_171/BiasAdd/ReadVariableOp2D
 conv2d_171/Conv2D/ReadVariableOp conv2d_171/Conv2D/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
 
_user_specified_nameinputs
	
Ó
8__inference_batch_normalization_159_layer_call_fn_423680

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_159_layer_call_and_return_conditional_losses_422025
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
×
8__inference_batch_normalization_162_layer_call_fn_423956

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_162_layer_call_and_return_conditional_losses_422253
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_168_layer_call_and_return_conditional_losses_423726

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_168_layer_call_and_return_conditional_losses_423654

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕk
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ×: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_169_layer_call_and_return_conditional_losses_422121

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
Â
S__inference_batch_normalization_158_layer_call_and_return_conditional_losses_421949

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_169_layer_call_and_return_conditional_losses_423746

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿFj: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFj
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_170_layer_call_and_return_conditional_losses_423838

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ"4 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"4 
 
_user_specified_nameinputs
Ü
Â
S__inference_batch_normalization_159_layer_call_and_return_conditional_losses_422025

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
Â
S__inference_batch_normalization_160_layer_call_and_return_conditional_losses_422101

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Î

S__inference_batch_normalization_161_layer_call_and_return_conditional_losses_423882

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Ó
8__inference_batch_normalization_159_layer_call_fn_423667

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_159_layer_call_and_return_conditional_losses_421994
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
Æ
S__inference_batch_normalization_162_layer_call_and_return_conditional_losses_423992

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ¾
$
!__inference__wrapped_model_421896
conv2d_167_inputQ
7sequential_21_conv2d_167_conv2d_readvariableop_resource:F
8sequential_21_conv2d_167_biasadd_readvariableop_resource:K
=sequential_21_batch_normalization_158_readvariableop_resource:M
?sequential_21_batch_normalization_158_readvariableop_1_resource:\
Nsequential_21_batch_normalization_158_fusedbatchnormv3_readvariableop_resource:^
Psequential_21_batch_normalization_158_fusedbatchnormv3_readvariableop_1_resource:Q
7sequential_21_conv2d_168_conv2d_readvariableop_resource:F
8sequential_21_conv2d_168_biasadd_readvariableop_resource:K
=sequential_21_batch_normalization_159_readvariableop_resource:M
?sequential_21_batch_normalization_159_readvariableop_1_resource:\
Nsequential_21_batch_normalization_159_fusedbatchnormv3_readvariableop_resource:^
Psequential_21_batch_normalization_159_fusedbatchnormv3_readvariableop_1_resource:Q
7sequential_21_conv2d_169_conv2d_readvariableop_resource: F
8sequential_21_conv2d_169_biasadd_readvariableop_resource: K
=sequential_21_batch_normalization_160_readvariableop_resource: M
?sequential_21_batch_normalization_160_readvariableop_1_resource: \
Nsequential_21_batch_normalization_160_fusedbatchnormv3_readvariableop_resource: ^
Psequential_21_batch_normalization_160_fusedbatchnormv3_readvariableop_1_resource: Q
7sequential_21_conv2d_170_conv2d_readvariableop_resource: @F
8sequential_21_conv2d_170_biasadd_readvariableop_resource:@K
=sequential_21_batch_normalization_161_readvariableop_resource:@M
?sequential_21_batch_normalization_161_readvariableop_1_resource:@\
Nsequential_21_batch_normalization_161_fusedbatchnormv3_readvariableop_resource:@^
Psequential_21_batch_normalization_161_fusedbatchnormv3_readvariableop_1_resource:@R
7sequential_21_conv2d_171_conv2d_readvariableop_resource:@G
8sequential_21_conv2d_171_biasadd_readvariableop_resource:	L
=sequential_21_batch_normalization_162_readvariableop_resource:	N
?sequential_21_batch_normalization_162_readvariableop_1_resource:	]
Nsequential_21_batch_normalization_162_fusedbatchnormv3_readvariableop_resource:	_
Psequential_21_batch_normalization_162_fusedbatchnormv3_readvariableop_1_resource:	H
5sequential_21_dense_37_matmul_readvariableop_resource:	M
D
6sequential_21_dense_37_biasadd_readvariableop_resource:

identity¢Esequential_21/batch_normalization_158/FusedBatchNormV3/ReadVariableOp¢Gsequential_21/batch_normalization_158/FusedBatchNormV3/ReadVariableOp_1¢4sequential_21/batch_normalization_158/ReadVariableOp¢6sequential_21/batch_normalization_158/ReadVariableOp_1¢Esequential_21/batch_normalization_159/FusedBatchNormV3/ReadVariableOp¢Gsequential_21/batch_normalization_159/FusedBatchNormV3/ReadVariableOp_1¢4sequential_21/batch_normalization_159/ReadVariableOp¢6sequential_21/batch_normalization_159/ReadVariableOp_1¢Esequential_21/batch_normalization_160/FusedBatchNormV3/ReadVariableOp¢Gsequential_21/batch_normalization_160/FusedBatchNormV3/ReadVariableOp_1¢4sequential_21/batch_normalization_160/ReadVariableOp¢6sequential_21/batch_normalization_160/ReadVariableOp_1¢Esequential_21/batch_normalization_161/FusedBatchNormV3/ReadVariableOp¢Gsequential_21/batch_normalization_161/FusedBatchNormV3/ReadVariableOp_1¢4sequential_21/batch_normalization_161/ReadVariableOp¢6sequential_21/batch_normalization_161/ReadVariableOp_1¢Esequential_21/batch_normalization_162/FusedBatchNormV3/ReadVariableOp¢Gsequential_21/batch_normalization_162/FusedBatchNormV3/ReadVariableOp_1¢4sequential_21/batch_normalization_162/ReadVariableOp¢6sequential_21/batch_normalization_162/ReadVariableOp_1¢/sequential_21/conv2d_167/BiasAdd/ReadVariableOp¢.sequential_21/conv2d_167/Conv2D/ReadVariableOp¢/sequential_21/conv2d_168/BiasAdd/ReadVariableOp¢.sequential_21/conv2d_168/Conv2D/ReadVariableOp¢/sequential_21/conv2d_169/BiasAdd/ReadVariableOp¢.sequential_21/conv2d_169/Conv2D/ReadVariableOp¢/sequential_21/conv2d_170/BiasAdd/ReadVariableOp¢.sequential_21/conv2d_170/Conv2D/ReadVariableOp¢/sequential_21/conv2d_171/BiasAdd/ReadVariableOp¢.sequential_21/conv2d_171/Conv2D/ReadVariableOp¢-sequential_21/dense_37/BiasAdd/ReadVariableOp¢,sequential_21/dense_37/MatMul/ReadVariableOp®
.sequential_21/conv2d_167/Conv2D/ReadVariableOpReadVariableOp7sequential_21_conv2d_167_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ø
sequential_21/conv2d_167/Conv2DConv2Dconv2d_167_input6sequential_21/conv2d_167/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®*
paddingVALID*
strides
¤
/sequential_21/conv2d_167/BiasAdd/ReadVariableOpReadVariableOp8sequential_21_conv2d_167_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ê
 sequential_21/conv2d_167/BiasAddBiasAdd(sequential_21/conv2d_167/Conv2D:output:07sequential_21/conv2d_167/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
sequential_21/conv2d_167/ReluRelu)sequential_21/conv2d_167/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®®
4sequential_21/batch_normalization_158/ReadVariableOpReadVariableOp=sequential_21_batch_normalization_158_readvariableop_resource*
_output_shapes
:*
dtype0²
6sequential_21/batch_normalization_158/ReadVariableOp_1ReadVariableOp?sequential_21_batch_normalization_158_readvariableop_1_resource*
_output_shapes
:*
dtype0Ð
Esequential_21/batch_normalization_158/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_21_batch_normalization_158_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ô
Gsequential_21/batch_normalization_158/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_21_batch_normalization_158_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
6sequential_21/batch_normalization_158/FusedBatchNormV3FusedBatchNormV3+sequential_21/conv2d_167/Relu:activations:0<sequential_21/batch_normalization_158/ReadVariableOp:value:0>sequential_21/batch_normalization_158/ReadVariableOp_1:value:0Msequential_21/batch_normalization_158/FusedBatchNormV3/ReadVariableOp:value:0Osequential_21/batch_normalization_158/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ®:::::*
epsilon%o:*
is_training( Ý
'sequential_21/max_pooling2d_167/MaxPoolMaxPool:sequential_21/batch_normalization_158/FusedBatchNormV3:y:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*
ksize
*
paddingVALID*
strides
®
.sequential_21/conv2d_168/Conv2D/ReadVariableOpReadVariableOp7sequential_21_conv2d_168_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ø
sequential_21/conv2d_168/Conv2DConv2D0sequential_21/max_pooling2d_167/MaxPool:output:06sequential_21/conv2d_168/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ*
paddingVALID*
strides
¤
/sequential_21/conv2d_168/BiasAdd/ReadVariableOpReadVariableOp8sequential_21_conv2d_168_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ê
 sequential_21/conv2d_168/BiasAddBiasAdd(sequential_21/conv2d_168/Conv2D:output:07sequential_21/conv2d_168/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
sequential_21/conv2d_168/ReluRelu)sequential_21/conv2d_168/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ®
4sequential_21/batch_normalization_159/ReadVariableOpReadVariableOp=sequential_21_batch_normalization_159_readvariableop_resource*
_output_shapes
:*
dtype0²
6sequential_21/batch_normalization_159/ReadVariableOp_1ReadVariableOp?sequential_21_batch_normalization_159_readvariableop_1_resource*
_output_shapes
:*
dtype0Ð
Esequential_21/batch_normalization_159/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_21_batch_normalization_159_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ô
Gsequential_21/batch_normalization_159/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_21_batch_normalization_159_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
6sequential_21/batch_normalization_159/FusedBatchNormV3FusedBatchNormV3+sequential_21/conv2d_168/Relu:activations:0<sequential_21/batch_normalization_159/ReadVariableOp:value:0>sequential_21/batch_normalization_159/ReadVariableOp_1:value:0Msequential_21/batch_normalization_159/FusedBatchNormV3/ReadVariableOp:value:0Osequential_21/batch_normalization_159/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿÕ:::::*
epsilon%o:*
is_training( Û
'sequential_21/max_pooling2d_168/MaxPoolMaxPool:sequential_21/batch_normalization_159/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFj*
ksize
*
paddingVALID*
strides
®
.sequential_21/conv2d_169/Conv2D/ReadVariableOpReadVariableOp7sequential_21_conv2d_169_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ö
sequential_21/conv2d_169/Conv2DConv2D0sequential_21/max_pooling2d_168/MaxPool:output:06sequential_21/conv2d_169/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh *
paddingVALID*
strides
¤
/sequential_21/conv2d_169/BiasAdd/ReadVariableOpReadVariableOp8sequential_21_conv2d_169_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0È
 sequential_21/conv2d_169/BiasAddBiasAdd(sequential_21/conv2d_169/Conv2D:output:07sequential_21/conv2d_169/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh 
sequential_21/conv2d_169/ReluRelu)sequential_21/conv2d_169/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh ®
4sequential_21/batch_normalization_160/ReadVariableOpReadVariableOp=sequential_21_batch_normalization_160_readvariableop_resource*
_output_shapes
: *
dtype0²
6sequential_21/batch_normalization_160/ReadVariableOp_1ReadVariableOp?sequential_21_batch_normalization_160_readvariableop_1_resource*
_output_shapes
: *
dtype0Ð
Esequential_21/batch_normalization_160/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_21_batch_normalization_160_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ô
Gsequential_21/batch_normalization_160/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_21_batch_normalization_160_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0
6sequential_21/batch_normalization_160/FusedBatchNormV3FusedBatchNormV3+sequential_21/conv2d_169/Relu:activations:0<sequential_21/batch_normalization_160/ReadVariableOp:value:0>sequential_21/batch_normalization_160/ReadVariableOp_1:value:0Msequential_21/batch_normalization_160/FusedBatchNormV3/ReadVariableOp:value:0Osequential_21/batch_normalization_160/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿDh : : : : :*
epsilon%o:*
is_training( Û
'sequential_21/max_pooling2d_169/MaxPoolMaxPool:sequential_21/batch_normalization_160/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"4 *
ksize
*
paddingVALID*
strides
®
.sequential_21/conv2d_170/Conv2D/ReadVariableOpReadVariableOp7sequential_21_conv2d_170_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0ö
sequential_21/conv2d_170/Conv2DConv2D0sequential_21/max_pooling2d_169/MaxPool:output:06sequential_21/conv2d_170/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@*
paddingVALID*
strides
¤
/sequential_21/conv2d_170/BiasAdd/ReadVariableOpReadVariableOp8sequential_21_conv2d_170_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0È
 sequential_21/conv2d_170/BiasAddBiasAdd(sequential_21/conv2d_170/Conv2D:output:07sequential_21/conv2d_170/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@
sequential_21/conv2d_170/ReluRelu)sequential_21/conv2d_170/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@®
4sequential_21/batch_normalization_161/ReadVariableOpReadVariableOp=sequential_21_batch_normalization_161_readvariableop_resource*
_output_shapes
:@*
dtype0²
6sequential_21/batch_normalization_161/ReadVariableOp_1ReadVariableOp?sequential_21_batch_normalization_161_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ð
Esequential_21/batch_normalization_161/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_21_batch_normalization_161_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ô
Gsequential_21/batch_normalization_161/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_21_batch_normalization_161_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
6sequential_21/batch_normalization_161/FusedBatchNormV3FusedBatchNormV3+sequential_21/conv2d_170/Relu:activations:0<sequential_21/batch_normalization_161/ReadVariableOp:value:0>sequential_21/batch_normalization_161/ReadVariableOp_1:value:0Msequential_21/batch_normalization_161/FusedBatchNormV3/ReadVariableOp:value:0Osequential_21/batch_normalization_161/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ 2@:@:@:@:@:*
epsilon%o:*
is_training( Û
'sequential_21/max_pooling2d_170/MaxPoolMaxPool:sequential_21/batch_normalization_161/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
¯
.sequential_21/conv2d_171/Conv2D/ReadVariableOpReadVariableOp7sequential_21_conv2d_171_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0÷
sequential_21/conv2d_171/Conv2DConv2D0sequential_21/max_pooling2d_170/MaxPool:output:06sequential_21/conv2d_171/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
¥
/sequential_21/conv2d_171/BiasAdd/ReadVariableOpReadVariableOp8sequential_21_conv2d_171_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0É
 sequential_21/conv2d_171/BiasAddBiasAdd(sequential_21/conv2d_171/Conv2D:output:07sequential_21/conv2d_171/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_21/conv2d_171/ReluRelu)sequential_21/conv2d_171/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
4sequential_21/batch_normalization_162/ReadVariableOpReadVariableOp=sequential_21_batch_normalization_162_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_21/batch_normalization_162/ReadVariableOp_1ReadVariableOp?sequential_21_batch_normalization_162_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ñ
Esequential_21/batch_normalization_162/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_21_batch_normalization_162_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
Gsequential_21/batch_normalization_162/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_21_batch_normalization_162_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
6sequential_21/batch_normalization_162/FusedBatchNormV3FusedBatchNormV3+sequential_21/conv2d_171/Relu:activations:0<sequential_21/batch_normalization_162/ReadVariableOp:value:0>sequential_21/batch_normalization_162/ReadVariableOp_1:value:0Msequential_21/batch_normalization_162/FusedBatchNormV3/ReadVariableOp:value:0Osequential_21/batch_normalization_162/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( Ü
'sequential_21/max_pooling2d_171/MaxPoolMaxPool:sequential_21/batch_normalization_162/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
o
sequential_21/flatten_38/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&  ¹
 sequential_21/flatten_38/ReshapeReshape0sequential_21/max_pooling2d_171/MaxPool:output:0'sequential_21/flatten_38/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
!sequential_21/dropout_40/IdentityIdentity)sequential_21/flatten_38/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM£
,sequential_21/dense_37/MatMul/ReadVariableOpReadVariableOp5sequential_21_dense_37_matmul_readvariableop_resource*
_output_shapes
:	M
*
dtype0»
sequential_21/dense_37/MatMulMatMul*sequential_21/dropout_40/Identity:output:04sequential_21/dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
-sequential_21/dense_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_37_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0»
sequential_21/dense_37/BiasAddBiasAdd'sequential_21/dense_37/MatMul:product:05sequential_21/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

sequential_21/dense_37/SoftmaxSoftmax'sequential_21/dense_37/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
IdentityIdentity(sequential_21/dense_37/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpF^sequential_21/batch_normalization_158/FusedBatchNormV3/ReadVariableOpH^sequential_21/batch_normalization_158/FusedBatchNormV3/ReadVariableOp_15^sequential_21/batch_normalization_158/ReadVariableOp7^sequential_21/batch_normalization_158/ReadVariableOp_1F^sequential_21/batch_normalization_159/FusedBatchNormV3/ReadVariableOpH^sequential_21/batch_normalization_159/FusedBatchNormV3/ReadVariableOp_15^sequential_21/batch_normalization_159/ReadVariableOp7^sequential_21/batch_normalization_159/ReadVariableOp_1F^sequential_21/batch_normalization_160/FusedBatchNormV3/ReadVariableOpH^sequential_21/batch_normalization_160/FusedBatchNormV3/ReadVariableOp_15^sequential_21/batch_normalization_160/ReadVariableOp7^sequential_21/batch_normalization_160/ReadVariableOp_1F^sequential_21/batch_normalization_161/FusedBatchNormV3/ReadVariableOpH^sequential_21/batch_normalization_161/FusedBatchNormV3/ReadVariableOp_15^sequential_21/batch_normalization_161/ReadVariableOp7^sequential_21/batch_normalization_161/ReadVariableOp_1F^sequential_21/batch_normalization_162/FusedBatchNormV3/ReadVariableOpH^sequential_21/batch_normalization_162/FusedBatchNormV3/ReadVariableOp_15^sequential_21/batch_normalization_162/ReadVariableOp7^sequential_21/batch_normalization_162/ReadVariableOp_10^sequential_21/conv2d_167/BiasAdd/ReadVariableOp/^sequential_21/conv2d_167/Conv2D/ReadVariableOp0^sequential_21/conv2d_168/BiasAdd/ReadVariableOp/^sequential_21/conv2d_168/Conv2D/ReadVariableOp0^sequential_21/conv2d_169/BiasAdd/ReadVariableOp/^sequential_21/conv2d_169/Conv2D/ReadVariableOp0^sequential_21/conv2d_170/BiasAdd/ReadVariableOp/^sequential_21/conv2d_170/Conv2D/ReadVariableOp0^sequential_21/conv2d_171/BiasAdd/ReadVariableOp/^sequential_21/conv2d_171/Conv2D/ReadVariableOp.^sequential_21/dense_37/BiasAdd/ReadVariableOp-^sequential_21/dense_37/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ °: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
Esequential_21/batch_normalization_158/FusedBatchNormV3/ReadVariableOpEsequential_21/batch_normalization_158/FusedBatchNormV3/ReadVariableOp2
Gsequential_21/batch_normalization_158/FusedBatchNormV3/ReadVariableOp_1Gsequential_21/batch_normalization_158/FusedBatchNormV3/ReadVariableOp_12l
4sequential_21/batch_normalization_158/ReadVariableOp4sequential_21/batch_normalization_158/ReadVariableOp2p
6sequential_21/batch_normalization_158/ReadVariableOp_16sequential_21/batch_normalization_158/ReadVariableOp_12
Esequential_21/batch_normalization_159/FusedBatchNormV3/ReadVariableOpEsequential_21/batch_normalization_159/FusedBatchNormV3/ReadVariableOp2
Gsequential_21/batch_normalization_159/FusedBatchNormV3/ReadVariableOp_1Gsequential_21/batch_normalization_159/FusedBatchNormV3/ReadVariableOp_12l
4sequential_21/batch_normalization_159/ReadVariableOp4sequential_21/batch_normalization_159/ReadVariableOp2p
6sequential_21/batch_normalization_159/ReadVariableOp_16sequential_21/batch_normalization_159/ReadVariableOp_12
Esequential_21/batch_normalization_160/FusedBatchNormV3/ReadVariableOpEsequential_21/batch_normalization_160/FusedBatchNormV3/ReadVariableOp2
Gsequential_21/batch_normalization_160/FusedBatchNormV3/ReadVariableOp_1Gsequential_21/batch_normalization_160/FusedBatchNormV3/ReadVariableOp_12l
4sequential_21/batch_normalization_160/ReadVariableOp4sequential_21/batch_normalization_160/ReadVariableOp2p
6sequential_21/batch_normalization_160/ReadVariableOp_16sequential_21/batch_normalization_160/ReadVariableOp_12
Esequential_21/batch_normalization_161/FusedBatchNormV3/ReadVariableOpEsequential_21/batch_normalization_161/FusedBatchNormV3/ReadVariableOp2
Gsequential_21/batch_normalization_161/FusedBatchNormV3/ReadVariableOp_1Gsequential_21/batch_normalization_161/FusedBatchNormV3/ReadVariableOp_12l
4sequential_21/batch_normalization_161/ReadVariableOp4sequential_21/batch_normalization_161/ReadVariableOp2p
6sequential_21/batch_normalization_161/ReadVariableOp_16sequential_21/batch_normalization_161/ReadVariableOp_12
Esequential_21/batch_normalization_162/FusedBatchNormV3/ReadVariableOpEsequential_21/batch_normalization_162/FusedBatchNormV3/ReadVariableOp2
Gsequential_21/batch_normalization_162/FusedBatchNormV3/ReadVariableOp_1Gsequential_21/batch_normalization_162/FusedBatchNormV3/ReadVariableOp_12l
4sequential_21/batch_normalization_162/ReadVariableOp4sequential_21/batch_normalization_162/ReadVariableOp2p
6sequential_21/batch_normalization_162/ReadVariableOp_16sequential_21/batch_normalization_162/ReadVariableOp_12b
/sequential_21/conv2d_167/BiasAdd/ReadVariableOp/sequential_21/conv2d_167/BiasAdd/ReadVariableOp2`
.sequential_21/conv2d_167/Conv2D/ReadVariableOp.sequential_21/conv2d_167/Conv2D/ReadVariableOp2b
/sequential_21/conv2d_168/BiasAdd/ReadVariableOp/sequential_21/conv2d_168/BiasAdd/ReadVariableOp2`
.sequential_21/conv2d_168/Conv2D/ReadVariableOp.sequential_21/conv2d_168/Conv2D/ReadVariableOp2b
/sequential_21/conv2d_169/BiasAdd/ReadVariableOp/sequential_21/conv2d_169/BiasAdd/ReadVariableOp2`
.sequential_21/conv2d_169/Conv2D/ReadVariableOp.sequential_21/conv2d_169/Conv2D/ReadVariableOp2b
/sequential_21/conv2d_170/BiasAdd/ReadVariableOp/sequential_21/conv2d_170/BiasAdd/ReadVariableOp2`
.sequential_21/conv2d_170/Conv2D/ReadVariableOp.sequential_21/conv2d_170/Conv2D/ReadVariableOp2b
/sequential_21/conv2d_171/BiasAdd/ReadVariableOp/sequential_21/conv2d_171/BiasAdd/ReadVariableOp2`
.sequential_21/conv2d_171/Conv2D/ReadVariableOp.sequential_21/conv2d_171/Conv2D/ReadVariableOp2^
-sequential_21/dense_37/BiasAdd/ReadVariableOp-sequential_21/dense_37/BiasAdd/ReadVariableOp2\
,sequential_21/dense_37/MatMul/ReadVariableOp,sequential_21/dense_37/MatMul/ReadVariableOp:c _
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
*
_user_specified_nameconv2d_167_input
ò
¢
+__inference_conv2d_171_layer_call_fn_423919

inputs"
unknown:@
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_171_layer_call_and_return_conditional_losses_422402x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ü	
e
F__inference_dropout_40_layer_call_and_return_conditional_losses_422548

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿMC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿMp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿMj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿMZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿM:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
 
_user_specified_nameinputs
Ê
b
F__inference_flatten_38_layer_call_and_return_conditional_losses_424013

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿMY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
¢
S__inference_batch_normalization_162_layer_call_and_return_conditional_losses_422222

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
G
+__inference_flatten_38_layer_call_fn_424007

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_38_layer_call_and_return_conditional_losses_422424a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
¢
S__inference_batch_normalization_162_layer_call_and_return_conditional_losses_423974

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ë
.__inference_sequential_21_layer_call_fn_422518
conv2d_167_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: @

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@%

unknown_23:@

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	M


unknown_30:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_167_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_21_layer_call_and_return_conditional_losses_422451o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ °: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
*
_user_specified_nameconv2d_167_input


F__inference_conv2d_171_layer_call_and_return_conditional_losses_423930

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ü
Â
S__inference_batch_normalization_159_layer_call_and_return_conditional_losses_423716

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î

S__inference_batch_normalization_158_layer_call_and_return_conditional_losses_421918

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î

S__inference_batch_normalization_160_layer_call_and_return_conditional_losses_423790

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
÷
d
+__inference_dropout_40_layer_call_fn_424023

inputs
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_40_layer_call_and_return_conditional_losses_422548p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿM22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
 
_user_specified_nameinputs
Å

)__inference_dense_37_layer_call_fn_424049

inputs
unknown:	M

	unknown_0:

identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_422444o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿM: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
 
_user_specified_nameinputs
î
 
+__inference_conv2d_169_layer_call_fn_423735

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_169_layer_call_and_return_conditional_losses_422348w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿFj: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFj
 
_user_specified_nameinputs
¼
N
2__inference_max_pooling2d_170_layer_call_fn_423905

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_170_layer_call_and_return_conditional_losses_422197
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

ö
D__inference_dense_37_layer_call_and_return_conditional_losses_422444

inputs1
matmul_readvariableop_resource:	M
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	M
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿM: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
 
_user_specified_nameinputs
Ü
Â
S__inference_batch_normalization_161_layer_call_and_return_conditional_losses_422177

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_171_layer_call_and_return_conditional_losses_424002

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_170_layer_call_and_return_conditional_losses_422197

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é
á
$__inference_signature_wrapper_423542
conv2d_167_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: @

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@%

unknown_23:@

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	M


unknown_30:

identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallconv2d_167_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_421896o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ °: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
*
_user_specified_nameconv2d_167_input
åÙ
¤9
"__inference__traced_restore_424603
file_prefix<
"assignvariableop_conv2d_167_kernel:0
"assignvariableop_1_conv2d_167_bias:>
0assignvariableop_2_batch_normalization_158_gamma:=
/assignvariableop_3_batch_normalization_158_beta:D
6assignvariableop_4_batch_normalization_158_moving_mean:H
:assignvariableop_5_batch_normalization_158_moving_variance:>
$assignvariableop_6_conv2d_168_kernel:0
"assignvariableop_7_conv2d_168_bias:>
0assignvariableop_8_batch_normalization_159_gamma:=
/assignvariableop_9_batch_normalization_159_beta:E
7assignvariableop_10_batch_normalization_159_moving_mean:I
;assignvariableop_11_batch_normalization_159_moving_variance:?
%assignvariableop_12_conv2d_169_kernel: 1
#assignvariableop_13_conv2d_169_bias: ?
1assignvariableop_14_batch_normalization_160_gamma: >
0assignvariableop_15_batch_normalization_160_beta: E
7assignvariableop_16_batch_normalization_160_moving_mean: I
;assignvariableop_17_batch_normalization_160_moving_variance: ?
%assignvariableop_18_conv2d_170_kernel: @1
#assignvariableop_19_conv2d_170_bias:@?
1assignvariableop_20_batch_normalization_161_gamma:@>
0assignvariableop_21_batch_normalization_161_beta:@E
7assignvariableop_22_batch_normalization_161_moving_mean:@I
;assignvariableop_23_batch_normalization_161_moving_variance:@@
%assignvariableop_24_conv2d_171_kernel:@2
#assignvariableop_25_conv2d_171_bias:	@
1assignvariableop_26_batch_normalization_162_gamma:	?
0assignvariableop_27_batch_normalization_162_beta:	F
7assignvariableop_28_batch_normalization_162_moving_mean:	J
;assignvariableop_29_batch_normalization_162_moving_variance:	6
#assignvariableop_30_dense_37_kernel:	M
/
!assignvariableop_31_dense_37_bias:
'
assignvariableop_32_adam_iter:	 )
assignvariableop_33_adam_beta_1: )
assignvariableop_34_adam_beta_2: (
assignvariableop_35_adam_decay: 0
&assignvariableop_36_adam_learning_rate: #
assignvariableop_37_total: #
assignvariableop_38_count: %
assignvariableop_39_total_1: %
assignvariableop_40_count_1: F
,assignvariableop_41_adam_conv2d_167_kernel_m:8
*assignvariableop_42_adam_conv2d_167_bias_m:F
8assignvariableop_43_adam_batch_normalization_158_gamma_m:E
7assignvariableop_44_adam_batch_normalization_158_beta_m:F
,assignvariableop_45_adam_conv2d_168_kernel_m:8
*assignvariableop_46_adam_conv2d_168_bias_m:F
8assignvariableop_47_adam_batch_normalization_159_gamma_m:E
7assignvariableop_48_adam_batch_normalization_159_beta_m:F
,assignvariableop_49_adam_conv2d_169_kernel_m: 8
*assignvariableop_50_adam_conv2d_169_bias_m: F
8assignvariableop_51_adam_batch_normalization_160_gamma_m: E
7assignvariableop_52_adam_batch_normalization_160_beta_m: F
,assignvariableop_53_adam_conv2d_170_kernel_m: @8
*assignvariableop_54_adam_conv2d_170_bias_m:@F
8assignvariableop_55_adam_batch_normalization_161_gamma_m:@E
7assignvariableop_56_adam_batch_normalization_161_beta_m:@G
,assignvariableop_57_adam_conv2d_171_kernel_m:@9
*assignvariableop_58_adam_conv2d_171_bias_m:	G
8assignvariableop_59_adam_batch_normalization_162_gamma_m:	F
7assignvariableop_60_adam_batch_normalization_162_beta_m:	=
*assignvariableop_61_adam_dense_37_kernel_m:	M
6
(assignvariableop_62_adam_dense_37_bias_m:
F
,assignvariableop_63_adam_conv2d_167_kernel_v:8
*assignvariableop_64_adam_conv2d_167_bias_v:F
8assignvariableop_65_adam_batch_normalization_158_gamma_v:E
7assignvariableop_66_adam_batch_normalization_158_beta_v:F
,assignvariableop_67_adam_conv2d_168_kernel_v:8
*assignvariableop_68_adam_conv2d_168_bias_v:F
8assignvariableop_69_adam_batch_normalization_159_gamma_v:E
7assignvariableop_70_adam_batch_normalization_159_beta_v:F
,assignvariableop_71_adam_conv2d_169_kernel_v: 8
*assignvariableop_72_adam_conv2d_169_bias_v: F
8assignvariableop_73_adam_batch_normalization_160_gamma_v: E
7assignvariableop_74_adam_batch_normalization_160_beta_v: F
,assignvariableop_75_adam_conv2d_170_kernel_v: @8
*assignvariableop_76_adam_conv2d_170_bias_v:@F
8assignvariableop_77_adam_batch_normalization_161_gamma_v:@E
7assignvariableop_78_adam_batch_normalization_161_beta_v:@G
,assignvariableop_79_adam_conv2d_171_kernel_v:@9
*assignvariableop_80_adam_conv2d_171_bias_v:	G
8assignvariableop_81_adam_batch_normalization_162_gamma_v:	F
7assignvariableop_82_adam_batch_normalization_162_beta_v:	=
*assignvariableop_83_adam_dense_37_kernel_v:	M
6
(assignvariableop_84_adam_dense_37_bias_v:

identity_86¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_9µ/
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*Û.
valueÑ.BÎ.VB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*Á
value·B´VB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ï
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*î
_output_shapesÛ
Ø::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*d
dtypesZ
X2V	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_167_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_167_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp0assignvariableop_2_batch_normalization_158_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp/assignvariableop_3_batch_normalization_158_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_4AssignVariableOp6assignvariableop_4_batch_normalization_158_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_5AssignVariableOp:assignvariableop_5_batch_normalization_158_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp$assignvariableop_6_conv2d_168_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_168_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp0assignvariableop_8_batch_normalization_159_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp/assignvariableop_9_batch_normalization_159_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_10AssignVariableOp7assignvariableop_10_batch_normalization_159_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_11AssignVariableOp;assignvariableop_11_batch_normalization_159_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp%assignvariableop_12_conv2d_169_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_169_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_14AssignVariableOp1assignvariableop_14_batch_normalization_160_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_15AssignVariableOp0assignvariableop_15_batch_normalization_160_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_16AssignVariableOp7assignvariableop_16_batch_normalization_160_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_17AssignVariableOp;assignvariableop_17_batch_normalization_160_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp%assignvariableop_18_conv2d_170_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp#assignvariableop_19_conv2d_170_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_20AssignVariableOp1assignvariableop_20_batch_normalization_161_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_21AssignVariableOp0assignvariableop_21_batch_normalization_161_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_22AssignVariableOp7assignvariableop_22_batch_normalization_161_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_23AssignVariableOp;assignvariableop_23_batch_normalization_161_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp%assignvariableop_24_conv2d_171_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp#assignvariableop_25_conv2d_171_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_26AssignVariableOp1assignvariableop_26_batch_normalization_162_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_27AssignVariableOp0assignvariableop_27_batch_normalization_162_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_28AssignVariableOp7assignvariableop_28_batch_normalization_162_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_29AssignVariableOp;assignvariableop_29_batch_normalization_162_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp#assignvariableop_30_dense_37_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp!assignvariableop_31_dense_37_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_iterIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_beta_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOpassignvariableop_34_adam_beta_2Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOpassignvariableop_35_adam_decayIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_learning_rateIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOpassignvariableop_37_totalIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOpassignvariableop_38_countIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOpassignvariableop_39_total_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOpassignvariableop_40_count_1Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp,assignvariableop_41_adam_conv2d_167_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_conv2d_167_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_43AssignVariableOp8assignvariableop_43_adam_batch_normalization_158_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_44AssignVariableOp7assignvariableop_44_adam_batch_normalization_158_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp,assignvariableop_45_adam_conv2d_168_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_conv2d_168_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_47AssignVariableOp8assignvariableop_47_adam_batch_normalization_159_gamma_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_48AssignVariableOp7assignvariableop_48_adam_batch_normalization_159_beta_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp,assignvariableop_49_adam_conv2d_169_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_conv2d_169_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_51AssignVariableOp8assignvariableop_51_adam_batch_normalization_160_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_52AssignVariableOp7assignvariableop_52_adam_batch_normalization_160_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp,assignvariableop_53_adam_conv2d_170_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_conv2d_170_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_55AssignVariableOp8assignvariableop_55_adam_batch_normalization_161_gamma_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_56AssignVariableOp7assignvariableop_56_adam_batch_normalization_161_beta_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp,assignvariableop_57_adam_conv2d_171_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_conv2d_171_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_59AssignVariableOp8assignvariableop_59_adam_batch_normalization_162_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_60AssignVariableOp7assignvariableop_60_adam_batch_normalization_162_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_37_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_37_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOp,assignvariableop_63_adam_conv2d_167_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_conv2d_167_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_65AssignVariableOp8assignvariableop_65_adam_batch_normalization_158_gamma_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_66AssignVariableOp7assignvariableop_66_adam_batch_normalization_158_beta_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_67AssignVariableOp,assignvariableop_67_adam_conv2d_168_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_68AssignVariableOp*assignvariableop_68_adam_conv2d_168_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_69AssignVariableOp8assignvariableop_69_adam_batch_normalization_159_gamma_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_70AssignVariableOp7assignvariableop_70_adam_batch_normalization_159_beta_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_71AssignVariableOp,assignvariableop_71_adam_conv2d_169_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_conv2d_169_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_73AssignVariableOp8assignvariableop_73_adam_batch_normalization_160_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_74AssignVariableOp7assignvariableop_74_adam_batch_normalization_160_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_75AssignVariableOp,assignvariableop_75_adam_conv2d_170_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_76AssignVariableOp*assignvariableop_76_adam_conv2d_170_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_77AssignVariableOp8assignvariableop_77_adam_batch_normalization_161_gamma_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_78AssignVariableOp7assignvariableop_78_adam_batch_normalization_161_beta_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_79AssignVariableOp,assignvariableop_79_adam_conv2d_171_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_80AssignVariableOp*assignvariableop_80_adam_conv2d_171_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_81AssignVariableOp8assignvariableop_81_adam_batch_normalization_162_gamma_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_82AssignVariableOp7assignvariableop_82_adam_batch_normalization_162_beta_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_83AssignVariableOp*assignvariableop_83_adam_dense_37_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_84AssignVariableOp(assignvariableop_84_adam_dense_37_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_85Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_86IdentityIdentity_85:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_86Identity_86:output:0*Á
_input_shapes¯
¬: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
	
Ó
8__inference_batch_normalization_158_layer_call_fn_423588

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_158_layer_call_and_return_conditional_losses_421949
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_sequential_21_layer_call_and_return_conditional_losses_423340

inputsC
)conv2d_167_conv2d_readvariableop_resource:8
*conv2d_167_biasadd_readvariableop_resource:=
/batch_normalization_158_readvariableop_resource:?
1batch_normalization_158_readvariableop_1_resource:N
@batch_normalization_158_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_158_fusedbatchnormv3_readvariableop_1_resource:C
)conv2d_168_conv2d_readvariableop_resource:8
*conv2d_168_biasadd_readvariableop_resource:=
/batch_normalization_159_readvariableop_resource:?
1batch_normalization_159_readvariableop_1_resource:N
@batch_normalization_159_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_159_fusedbatchnormv3_readvariableop_1_resource:C
)conv2d_169_conv2d_readvariableop_resource: 8
*conv2d_169_biasadd_readvariableop_resource: =
/batch_normalization_160_readvariableop_resource: ?
1batch_normalization_160_readvariableop_1_resource: N
@batch_normalization_160_fusedbatchnormv3_readvariableop_resource: P
Bbatch_normalization_160_fusedbatchnormv3_readvariableop_1_resource: C
)conv2d_170_conv2d_readvariableop_resource: @8
*conv2d_170_biasadd_readvariableop_resource:@=
/batch_normalization_161_readvariableop_resource:@?
1batch_normalization_161_readvariableop_1_resource:@N
@batch_normalization_161_fusedbatchnormv3_readvariableop_resource:@P
Bbatch_normalization_161_fusedbatchnormv3_readvariableop_1_resource:@D
)conv2d_171_conv2d_readvariableop_resource:@9
*conv2d_171_biasadd_readvariableop_resource:	>
/batch_normalization_162_readvariableop_resource:	@
1batch_normalization_162_readvariableop_1_resource:	O
@batch_normalization_162_fusedbatchnormv3_readvariableop_resource:	Q
Bbatch_normalization_162_fusedbatchnormv3_readvariableop_1_resource:	:
'dense_37_matmul_readvariableop_resource:	M
6
(dense_37_biasadd_readvariableop_resource:

identity¢7batch_normalization_158/FusedBatchNormV3/ReadVariableOp¢9batch_normalization_158/FusedBatchNormV3/ReadVariableOp_1¢&batch_normalization_158/ReadVariableOp¢(batch_normalization_158/ReadVariableOp_1¢7batch_normalization_159/FusedBatchNormV3/ReadVariableOp¢9batch_normalization_159/FusedBatchNormV3/ReadVariableOp_1¢&batch_normalization_159/ReadVariableOp¢(batch_normalization_159/ReadVariableOp_1¢7batch_normalization_160/FusedBatchNormV3/ReadVariableOp¢9batch_normalization_160/FusedBatchNormV3/ReadVariableOp_1¢&batch_normalization_160/ReadVariableOp¢(batch_normalization_160/ReadVariableOp_1¢7batch_normalization_161/FusedBatchNormV3/ReadVariableOp¢9batch_normalization_161/FusedBatchNormV3/ReadVariableOp_1¢&batch_normalization_161/ReadVariableOp¢(batch_normalization_161/ReadVariableOp_1¢7batch_normalization_162/FusedBatchNormV3/ReadVariableOp¢9batch_normalization_162/FusedBatchNormV3/ReadVariableOp_1¢&batch_normalization_162/ReadVariableOp¢(batch_normalization_162/ReadVariableOp_1¢!conv2d_167/BiasAdd/ReadVariableOp¢ conv2d_167/Conv2D/ReadVariableOp¢!conv2d_168/BiasAdd/ReadVariableOp¢ conv2d_168/Conv2D/ReadVariableOp¢!conv2d_169/BiasAdd/ReadVariableOp¢ conv2d_169/Conv2D/ReadVariableOp¢!conv2d_170/BiasAdd/ReadVariableOp¢ conv2d_170/Conv2D/ReadVariableOp¢!conv2d_171/BiasAdd/ReadVariableOp¢ conv2d_171/Conv2D/ReadVariableOp¢dense_37/BiasAdd/ReadVariableOp¢dense_37/MatMul/ReadVariableOp
 conv2d_167/Conv2D/ReadVariableOpReadVariableOp)conv2d_167_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0²
conv2d_167/Conv2DConv2Dinputs(conv2d_167/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®*
paddingVALID*
strides

!conv2d_167/BiasAdd/ReadVariableOpReadVariableOp*conv2d_167_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
conv2d_167/BiasAddBiasAddconv2d_167/Conv2D:output:0)conv2d_167/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®p
conv2d_167/ReluReluconv2d_167/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
&batch_normalization_158/ReadVariableOpReadVariableOp/batch_normalization_158_readvariableop_resource*
_output_shapes
:*
dtype0
(batch_normalization_158/ReadVariableOp_1ReadVariableOp1batch_normalization_158_readvariableop_1_resource*
_output_shapes
:*
dtype0´
7batch_normalization_158/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_158_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¸
9batch_normalization_158/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_158_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ç
(batch_normalization_158/FusedBatchNormV3FusedBatchNormV3conv2d_167/Relu:activations:0.batch_normalization_158/ReadVariableOp:value:00batch_normalization_158/ReadVariableOp_1:value:0?batch_normalization_158/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_158/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ®:::::*
epsilon%o:*
is_training( Á
max_pooling2d_167/MaxPoolMaxPool,batch_normalization_158/FusedBatchNormV3:y:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*
ksize
*
paddingVALID*
strides

 conv2d_168/Conv2D/ReadVariableOpReadVariableOp)conv2d_168_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Î
conv2d_168/Conv2DConv2D"max_pooling2d_167/MaxPool:output:0(conv2d_168/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ*
paddingVALID*
strides

!conv2d_168/BiasAdd/ReadVariableOpReadVariableOp*conv2d_168_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
conv2d_168/BiasAddBiasAddconv2d_168/Conv2D:output:0)conv2d_168/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕp
conv2d_168/ReluReluconv2d_168/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
&batch_normalization_159/ReadVariableOpReadVariableOp/batch_normalization_159_readvariableop_resource*
_output_shapes
:*
dtype0
(batch_normalization_159/ReadVariableOp_1ReadVariableOp1batch_normalization_159_readvariableop_1_resource*
_output_shapes
:*
dtype0´
7batch_normalization_159/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_159_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¸
9batch_normalization_159/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_159_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ç
(batch_normalization_159/FusedBatchNormV3FusedBatchNormV3conv2d_168/Relu:activations:0.batch_normalization_159/ReadVariableOp:value:00batch_normalization_159/ReadVariableOp_1:value:0?batch_normalization_159/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_159/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿÕ:::::*
epsilon%o:*
is_training( ¿
max_pooling2d_168/MaxPoolMaxPool,batch_normalization_159/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFj*
ksize
*
paddingVALID*
strides

 conv2d_169/Conv2D/ReadVariableOpReadVariableOp)conv2d_169_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ì
conv2d_169/Conv2DConv2D"max_pooling2d_168/MaxPool:output:0(conv2d_169/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh *
paddingVALID*
strides

!conv2d_169/BiasAdd/ReadVariableOpReadVariableOp*conv2d_169_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_169/BiasAddBiasAddconv2d_169/Conv2D:output:0)conv2d_169/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh n
conv2d_169/ReluReluconv2d_169/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh 
&batch_normalization_160/ReadVariableOpReadVariableOp/batch_normalization_160_readvariableop_resource*
_output_shapes
: *
dtype0
(batch_normalization_160/ReadVariableOp_1ReadVariableOp1batch_normalization_160_readvariableop_1_resource*
_output_shapes
: *
dtype0´
7batch_normalization_160/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_160_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¸
9batch_normalization_160/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_160_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Å
(batch_normalization_160/FusedBatchNormV3FusedBatchNormV3conv2d_169/Relu:activations:0.batch_normalization_160/ReadVariableOp:value:00batch_normalization_160/ReadVariableOp_1:value:0?batch_normalization_160/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_160/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿDh : : : : :*
epsilon%o:*
is_training( ¿
max_pooling2d_169/MaxPoolMaxPool,batch_normalization_160/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"4 *
ksize
*
paddingVALID*
strides

 conv2d_170/Conv2D/ReadVariableOpReadVariableOp)conv2d_170_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ì
conv2d_170/Conv2DConv2D"max_pooling2d_169/MaxPool:output:0(conv2d_170/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@*
paddingVALID*
strides

!conv2d_170/BiasAdd/ReadVariableOpReadVariableOp*conv2d_170_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_170/BiasAddBiasAddconv2d_170/Conv2D:output:0)conv2d_170/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@n
conv2d_170/ReluReluconv2d_170/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@
&batch_normalization_161/ReadVariableOpReadVariableOp/batch_normalization_161_readvariableop_resource*
_output_shapes
:@*
dtype0
(batch_normalization_161/ReadVariableOp_1ReadVariableOp1batch_normalization_161_readvariableop_1_resource*
_output_shapes
:@*
dtype0´
7batch_normalization_161/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_161_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¸
9batch_normalization_161/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_161_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Å
(batch_normalization_161/FusedBatchNormV3FusedBatchNormV3conv2d_170/Relu:activations:0.batch_normalization_161/ReadVariableOp:value:00batch_normalization_161/ReadVariableOp_1:value:0?batch_normalization_161/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_161/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ 2@:@:@:@:@:*
epsilon%o:*
is_training( ¿
max_pooling2d_170/MaxPoolMaxPool,batch_normalization_161/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

 conv2d_171/Conv2D/ReadVariableOpReadVariableOp)conv2d_171_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Í
conv2d_171/Conv2DConv2D"max_pooling2d_170/MaxPool:output:0(conv2d_171/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

!conv2d_171/BiasAdd/ReadVariableOpReadVariableOp*conv2d_171_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_171/BiasAddBiasAddconv2d_171/Conv2D:output:0)conv2d_171/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
conv2d_171/ReluReluconv2d_171/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&batch_normalization_162/ReadVariableOpReadVariableOp/batch_normalization_162_readvariableop_resource*
_output_shapes	
:*
dtype0
(batch_normalization_162/ReadVariableOp_1ReadVariableOp1batch_normalization_162_readvariableop_1_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_162/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_162_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0¹
9batch_normalization_162/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_162_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ê
(batch_normalization_162/FusedBatchNormV3FusedBatchNormV3conv2d_171/Relu:activations:0.batch_normalization_162/ReadVariableOp:value:00batch_normalization_162/ReadVariableOp_1:value:0?batch_normalization_162/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_162/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( À
max_pooling2d_171/MaxPoolMaxPool,batch_normalization_162/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
a
flatten_38/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&  
flatten_38/ReshapeReshape"max_pooling2d_171/MaxPool:output:0flatten_38/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿMo
dropout_40/IdentityIdentityflatten_38/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes
:	M
*
dtype0
dense_37/MatMulMatMuldropout_40/Identity:output:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
dense_37/SoftmaxSoftmaxdense_37/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
IdentityIdentitydense_37/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Þ
NoOpNoOp8^batch_normalization_158/FusedBatchNormV3/ReadVariableOp:^batch_normalization_158/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_158/ReadVariableOp)^batch_normalization_158/ReadVariableOp_18^batch_normalization_159/FusedBatchNormV3/ReadVariableOp:^batch_normalization_159/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_159/ReadVariableOp)^batch_normalization_159/ReadVariableOp_18^batch_normalization_160/FusedBatchNormV3/ReadVariableOp:^batch_normalization_160/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_160/ReadVariableOp)^batch_normalization_160/ReadVariableOp_18^batch_normalization_161/FusedBatchNormV3/ReadVariableOp:^batch_normalization_161/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_161/ReadVariableOp)^batch_normalization_161/ReadVariableOp_18^batch_normalization_162/FusedBatchNormV3/ReadVariableOp:^batch_normalization_162/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_162/ReadVariableOp)^batch_normalization_162/ReadVariableOp_1"^conv2d_167/BiasAdd/ReadVariableOp!^conv2d_167/Conv2D/ReadVariableOp"^conv2d_168/BiasAdd/ReadVariableOp!^conv2d_168/Conv2D/ReadVariableOp"^conv2d_169/BiasAdd/ReadVariableOp!^conv2d_169/Conv2D/ReadVariableOp"^conv2d_170/BiasAdd/ReadVariableOp!^conv2d_170/Conv2D/ReadVariableOp"^conv2d_171/BiasAdd/ReadVariableOp!^conv2d_171/Conv2D/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp^dense_37/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ °: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2r
7batch_normalization_158/FusedBatchNormV3/ReadVariableOp7batch_normalization_158/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_158/FusedBatchNormV3/ReadVariableOp_19batch_normalization_158/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_158/ReadVariableOp&batch_normalization_158/ReadVariableOp2T
(batch_normalization_158/ReadVariableOp_1(batch_normalization_158/ReadVariableOp_12r
7batch_normalization_159/FusedBatchNormV3/ReadVariableOp7batch_normalization_159/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_159/FusedBatchNormV3/ReadVariableOp_19batch_normalization_159/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_159/ReadVariableOp&batch_normalization_159/ReadVariableOp2T
(batch_normalization_159/ReadVariableOp_1(batch_normalization_159/ReadVariableOp_12r
7batch_normalization_160/FusedBatchNormV3/ReadVariableOp7batch_normalization_160/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_160/FusedBatchNormV3/ReadVariableOp_19batch_normalization_160/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_160/ReadVariableOp&batch_normalization_160/ReadVariableOp2T
(batch_normalization_160/ReadVariableOp_1(batch_normalization_160/ReadVariableOp_12r
7batch_normalization_161/FusedBatchNormV3/ReadVariableOp7batch_normalization_161/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_161/FusedBatchNormV3/ReadVariableOp_19batch_normalization_161/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_161/ReadVariableOp&batch_normalization_161/ReadVariableOp2T
(batch_normalization_161/ReadVariableOp_1(batch_normalization_161/ReadVariableOp_12r
7batch_normalization_162/FusedBatchNormV3/ReadVariableOp7batch_normalization_162/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_162/FusedBatchNormV3/ReadVariableOp_19batch_normalization_162/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_162/ReadVariableOp&batch_normalization_162/ReadVariableOp2T
(batch_normalization_162/ReadVariableOp_1(batch_normalization_162/ReadVariableOp_12F
!conv2d_167/BiasAdd/ReadVariableOp!conv2d_167/BiasAdd/ReadVariableOp2D
 conv2d_167/Conv2D/ReadVariableOp conv2d_167/Conv2D/ReadVariableOp2F
!conv2d_168/BiasAdd/ReadVariableOp!conv2d_168/BiasAdd/ReadVariableOp2D
 conv2d_168/Conv2D/ReadVariableOp conv2d_168/Conv2D/ReadVariableOp2F
!conv2d_169/BiasAdd/ReadVariableOp!conv2d_169/BiasAdd/ReadVariableOp2D
 conv2d_169/Conv2D/ReadVariableOp conv2d_169/Conv2D/ReadVariableOp2F
!conv2d_170/BiasAdd/ReadVariableOp!conv2d_170/BiasAdd/ReadVariableOp2D
 conv2d_170/Conv2D/ReadVariableOp conv2d_170/Conv2D/ReadVariableOp2F
!conv2d_171/BiasAdd/ReadVariableOp!conv2d_171/BiasAdd/ReadVariableOp2D
 conv2d_171/Conv2D/ReadVariableOp conv2d_171/Conv2D/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
 
_user_specified_nameinputs
¼
N
2__inference_max_pooling2d_168_layer_call_fn_423721

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_168_layer_call_and_return_conditional_losses_422045
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å\
Ê
I__inference_sequential_21_layer_call_and_return_conditional_losses_422764

inputs+
conv2d_167_422681:
conv2d_167_422683:,
batch_normalization_158_422686:,
batch_normalization_158_422688:,
batch_normalization_158_422690:,
batch_normalization_158_422692:+
conv2d_168_422696:
conv2d_168_422698:,
batch_normalization_159_422701:,
batch_normalization_159_422703:,
batch_normalization_159_422705:,
batch_normalization_159_422707:+
conv2d_169_422711: 
conv2d_169_422713: ,
batch_normalization_160_422716: ,
batch_normalization_160_422718: ,
batch_normalization_160_422720: ,
batch_normalization_160_422722: +
conv2d_170_422726: @
conv2d_170_422728:@,
batch_normalization_161_422731:@,
batch_normalization_161_422733:@,
batch_normalization_161_422735:@,
batch_normalization_161_422737:@,
conv2d_171_422741:@ 
conv2d_171_422743:	-
batch_normalization_162_422746:	-
batch_normalization_162_422748:	-
batch_normalization_162_422750:	-
batch_normalization_162_422752:	"
dense_37_422758:	M

dense_37_422760:

identity¢/batch_normalization_158/StatefulPartitionedCall¢/batch_normalization_159/StatefulPartitionedCall¢/batch_normalization_160/StatefulPartitionedCall¢/batch_normalization_161/StatefulPartitionedCall¢/batch_normalization_162/StatefulPartitionedCall¢"conv2d_167/StatefulPartitionedCall¢"conv2d_168/StatefulPartitionedCall¢"conv2d_169/StatefulPartitionedCall¢"conv2d_170/StatefulPartitionedCall¢"conv2d_171/StatefulPartitionedCall¢ dense_37/StatefulPartitionedCall¢"dropout_40/StatefulPartitionedCall
"conv2d_167/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_167_422681conv2d_167_422683*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_167_layer_call_and_return_conditional_losses_422294
/batch_normalization_158/StatefulPartitionedCallStatefulPartitionedCall+conv2d_167/StatefulPartitionedCall:output:0batch_normalization_158_422686batch_normalization_158_422688batch_normalization_158_422690batch_normalization_158_422692*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_158_layer_call_and_return_conditional_losses_421949
!max_pooling2d_167/PartitionedCallPartitionedCall8batch_normalization_158/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_167_layer_call_and_return_conditional_losses_421969¦
"conv2d_168/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_167/PartitionedCall:output:0conv2d_168_422696conv2d_168_422698*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_168_layer_call_and_return_conditional_losses_422321
/batch_normalization_159/StatefulPartitionedCallStatefulPartitionedCall+conv2d_168/StatefulPartitionedCall:output:0batch_normalization_159_422701batch_normalization_159_422703batch_normalization_159_422705batch_normalization_159_422707*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_159_layer_call_and_return_conditional_losses_422025
!max_pooling2d_168/PartitionedCallPartitionedCall8batch_normalization_159/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿFj* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_168_layer_call_and_return_conditional_losses_422045¤
"conv2d_169/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_168/PartitionedCall:output:0conv2d_169_422711conv2d_169_422713*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_169_layer_call_and_return_conditional_losses_422348
/batch_normalization_160/StatefulPartitionedCallStatefulPartitionedCall+conv2d_169/StatefulPartitionedCall:output:0batch_normalization_160_422716batch_normalization_160_422718batch_normalization_160_422720batch_normalization_160_422722*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_160_layer_call_and_return_conditional_losses_422101
!max_pooling2d_169/PartitionedCallPartitionedCall8batch_normalization_160/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"4 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_169_layer_call_and_return_conditional_losses_422121¤
"conv2d_170/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_169/PartitionedCall:output:0conv2d_170_422726conv2d_170_422728*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_170_layer_call_and_return_conditional_losses_422375
/batch_normalization_161/StatefulPartitionedCallStatefulPartitionedCall+conv2d_170/StatefulPartitionedCall:output:0batch_normalization_161_422731batch_normalization_161_422733batch_normalization_161_422735batch_normalization_161_422737*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_161_layer_call_and_return_conditional_losses_422177
!max_pooling2d_170/PartitionedCallPartitionedCall8batch_normalization_161/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_170_layer_call_and_return_conditional_losses_422197¥
"conv2d_171/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_170/PartitionedCall:output:0conv2d_171_422741conv2d_171_422743*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_171_layer_call_and_return_conditional_losses_422402
/batch_normalization_162/StatefulPartitionedCallStatefulPartitionedCall+conv2d_171/StatefulPartitionedCall:output:0batch_normalization_162_422746batch_normalization_162_422748batch_normalization_162_422750batch_normalization_162_422752*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_162_layer_call_and_return_conditional_losses_422253
!max_pooling2d_171/PartitionedCallPartitionedCall8batch_normalization_162/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_171_layer_call_and_return_conditional_losses_422273á
flatten_38/PartitionedCallPartitionedCall*max_pooling2d_171/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_38_layer_call_and_return_conditional_losses_422424ê
"dropout_40/StatefulPartitionedCallStatefulPartitionedCall#flatten_38/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_40_layer_call_and_return_conditional_losses_422548
 dense_37/StatefulPartitionedCallStatefulPartitionedCall+dropout_40/StatefulPartitionedCall:output:0dense_37_422758dense_37_422760*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_422444x
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Á
NoOpNoOp0^batch_normalization_158/StatefulPartitionedCall0^batch_normalization_159/StatefulPartitionedCall0^batch_normalization_160/StatefulPartitionedCall0^batch_normalization_161/StatefulPartitionedCall0^batch_normalization_162/StatefulPartitionedCall#^conv2d_167/StatefulPartitionedCall#^conv2d_168/StatefulPartitionedCall#^conv2d_169/StatefulPartitionedCall#^conv2d_170/StatefulPartitionedCall#^conv2d_171/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall#^dropout_40/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ °: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_158/StatefulPartitionedCall/batch_normalization_158/StatefulPartitionedCall2b
/batch_normalization_159/StatefulPartitionedCall/batch_normalization_159/StatefulPartitionedCall2b
/batch_normalization_160/StatefulPartitionedCall/batch_normalization_160/StatefulPartitionedCall2b
/batch_normalization_161/StatefulPartitionedCall/batch_normalization_161/StatefulPartitionedCall2b
/batch_normalization_162/StatefulPartitionedCall/batch_normalization_162/StatefulPartitionedCall2H
"conv2d_167/StatefulPartitionedCall"conv2d_167/StatefulPartitionedCall2H
"conv2d_168/StatefulPartitionedCall"conv2d_168/StatefulPartitionedCall2H
"conv2d_169/StatefulPartitionedCall"conv2d_169/StatefulPartitionedCall2H
"conv2d_170/StatefulPartitionedCall"conv2d_170/StatefulPartitionedCall2H
"conv2d_171/StatefulPartitionedCall"conv2d_171/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2H
"dropout_40/StatefulPartitionedCall"dropout_40/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
 
_user_specified_nameinputs
Ý
d
F__inference_dropout_40_layer_call_and_return_conditional_losses_422431

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿM:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
 
_user_specified_nameinputs
Î

S__inference_batch_normalization_159_layer_call_and_return_conditional_losses_421994

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î

S__inference_batch_normalization_160_layer_call_and_return_conditional_losses_422070

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ü
Â
S__inference_batch_normalization_158_layer_call_and_return_conditional_losses_423624

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î

S__inference_batch_normalization_158_layer_call_and_return_conditional_losses_423606

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_168_layer_call_and_return_conditional_losses_422045

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_169_layer_call_and_return_conditional_losses_423818

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_170_layer_call_and_return_conditional_losses_422375

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ"4 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"4 
 
_user_specified_nameinputs
¤

ö
D__inference_dense_37_layer_call_and_return_conditional_losses_424060

inputs1
matmul_readvariableop_resource:	M
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	M
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿM: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
 
_user_specified_nameinputs
Ü
Â
S__inference_batch_normalization_161_layer_call_and_return_conditional_losses_423900

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

ë
.__inference_sequential_21_layer_call_fn_422900
conv2d_167_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: @

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@%

unknown_23:@

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	M


unknown_30:

identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallconv2d_167_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*8
_read_only_resource_inputs
	
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_21_layer_call_and_return_conditional_losses_422764o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ °: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
*
_user_specified_nameconv2d_167_input
Î

S__inference_batch_normalization_159_layer_call_and_return_conditional_losses_423698

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
×
8__inference_batch_normalization_162_layer_call_fn_423943

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_162_layer_call_and_return_conditional_losses_422222
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ó
8__inference_batch_normalization_161_layer_call_fn_423851

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_161_layer_call_and_return_conditional_losses_422146
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Ó
8__inference_batch_normalization_160_layer_call_fn_423759

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_160_layer_call_and_return_conditional_losses_422070
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
Ó
8__inference_batch_normalization_161_layer_call_fn_423864

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_161_layer_call_and_return_conditional_losses_422177
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_167_layer_call_and_return_conditional_losses_422294

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ °: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_169_layer_call_and_return_conditional_losses_422348

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿFj: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFj
 
_user_specified_nameinputs


F__inference_conv2d_171_layer_call_and_return_conditional_losses_422402

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_171_layer_call_and_return_conditional_losses_422273

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
á
.__inference_sequential_21_layer_call_fn_423147

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: @

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@%

unknown_23:@

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	M


unknown_30:

identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_21_layer_call_and_return_conditional_losses_422451o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ °: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
 
_user_specified_nameinputs
ª
å'
__inference__traced_save_424338
file_prefix0
,savev2_conv2d_167_kernel_read_readvariableop.
*savev2_conv2d_167_bias_read_readvariableop<
8savev2_batch_normalization_158_gamma_read_readvariableop;
7savev2_batch_normalization_158_beta_read_readvariableopB
>savev2_batch_normalization_158_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_158_moving_variance_read_readvariableop0
,savev2_conv2d_168_kernel_read_readvariableop.
*savev2_conv2d_168_bias_read_readvariableop<
8savev2_batch_normalization_159_gamma_read_readvariableop;
7savev2_batch_normalization_159_beta_read_readvariableopB
>savev2_batch_normalization_159_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_159_moving_variance_read_readvariableop0
,savev2_conv2d_169_kernel_read_readvariableop.
*savev2_conv2d_169_bias_read_readvariableop<
8savev2_batch_normalization_160_gamma_read_readvariableop;
7savev2_batch_normalization_160_beta_read_readvariableopB
>savev2_batch_normalization_160_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_160_moving_variance_read_readvariableop0
,savev2_conv2d_170_kernel_read_readvariableop.
*savev2_conv2d_170_bias_read_readvariableop<
8savev2_batch_normalization_161_gamma_read_readvariableop;
7savev2_batch_normalization_161_beta_read_readvariableopB
>savev2_batch_normalization_161_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_161_moving_variance_read_readvariableop0
,savev2_conv2d_171_kernel_read_readvariableop.
*savev2_conv2d_171_bias_read_readvariableop<
8savev2_batch_normalization_162_gamma_read_readvariableop;
7savev2_batch_normalization_162_beta_read_readvariableopB
>savev2_batch_normalization_162_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_162_moving_variance_read_readvariableop.
*savev2_dense_37_kernel_read_readvariableop,
(savev2_dense_37_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_conv2d_167_kernel_m_read_readvariableop5
1savev2_adam_conv2d_167_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_158_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_158_beta_m_read_readvariableop7
3savev2_adam_conv2d_168_kernel_m_read_readvariableop5
1savev2_adam_conv2d_168_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_159_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_159_beta_m_read_readvariableop7
3savev2_adam_conv2d_169_kernel_m_read_readvariableop5
1savev2_adam_conv2d_169_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_160_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_160_beta_m_read_readvariableop7
3savev2_adam_conv2d_170_kernel_m_read_readvariableop5
1savev2_adam_conv2d_170_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_161_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_161_beta_m_read_readvariableop7
3savev2_adam_conv2d_171_kernel_m_read_readvariableop5
1savev2_adam_conv2d_171_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_162_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_162_beta_m_read_readvariableop5
1savev2_adam_dense_37_kernel_m_read_readvariableop3
/savev2_adam_dense_37_bias_m_read_readvariableop7
3savev2_adam_conv2d_167_kernel_v_read_readvariableop5
1savev2_adam_conv2d_167_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_158_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_158_beta_v_read_readvariableop7
3savev2_adam_conv2d_168_kernel_v_read_readvariableop5
1savev2_adam_conv2d_168_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_159_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_159_beta_v_read_readvariableop7
3savev2_adam_conv2d_169_kernel_v_read_readvariableop5
1savev2_adam_conv2d_169_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_160_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_160_beta_v_read_readvariableop7
3savev2_adam_conv2d_170_kernel_v_read_readvariableop5
1savev2_adam_conv2d_170_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_161_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_161_beta_v_read_readvariableop7
3savev2_adam_conv2d_171_kernel_v_read_readvariableop5
1savev2_adam_conv2d_171_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_162_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_162_beta_v_read_readvariableop5
1savev2_adam_dense_37_kernel_v_read_readvariableop3
/savev2_adam_dense_37_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ²/
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*Û.
valueÑ.BÎ.VB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*Á
value·B´VB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ª&
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_167_kernel_read_readvariableop*savev2_conv2d_167_bias_read_readvariableop8savev2_batch_normalization_158_gamma_read_readvariableop7savev2_batch_normalization_158_beta_read_readvariableop>savev2_batch_normalization_158_moving_mean_read_readvariableopBsavev2_batch_normalization_158_moving_variance_read_readvariableop,savev2_conv2d_168_kernel_read_readvariableop*savev2_conv2d_168_bias_read_readvariableop8savev2_batch_normalization_159_gamma_read_readvariableop7savev2_batch_normalization_159_beta_read_readvariableop>savev2_batch_normalization_159_moving_mean_read_readvariableopBsavev2_batch_normalization_159_moving_variance_read_readvariableop,savev2_conv2d_169_kernel_read_readvariableop*savev2_conv2d_169_bias_read_readvariableop8savev2_batch_normalization_160_gamma_read_readvariableop7savev2_batch_normalization_160_beta_read_readvariableop>savev2_batch_normalization_160_moving_mean_read_readvariableopBsavev2_batch_normalization_160_moving_variance_read_readvariableop,savev2_conv2d_170_kernel_read_readvariableop*savev2_conv2d_170_bias_read_readvariableop8savev2_batch_normalization_161_gamma_read_readvariableop7savev2_batch_normalization_161_beta_read_readvariableop>savev2_batch_normalization_161_moving_mean_read_readvariableopBsavev2_batch_normalization_161_moving_variance_read_readvariableop,savev2_conv2d_171_kernel_read_readvariableop*savev2_conv2d_171_bias_read_readvariableop8savev2_batch_normalization_162_gamma_read_readvariableop7savev2_batch_normalization_162_beta_read_readvariableop>savev2_batch_normalization_162_moving_mean_read_readvariableopBsavev2_batch_normalization_162_moving_variance_read_readvariableop*savev2_dense_37_kernel_read_readvariableop(savev2_dense_37_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_conv2d_167_kernel_m_read_readvariableop1savev2_adam_conv2d_167_bias_m_read_readvariableop?savev2_adam_batch_normalization_158_gamma_m_read_readvariableop>savev2_adam_batch_normalization_158_beta_m_read_readvariableop3savev2_adam_conv2d_168_kernel_m_read_readvariableop1savev2_adam_conv2d_168_bias_m_read_readvariableop?savev2_adam_batch_normalization_159_gamma_m_read_readvariableop>savev2_adam_batch_normalization_159_beta_m_read_readvariableop3savev2_adam_conv2d_169_kernel_m_read_readvariableop1savev2_adam_conv2d_169_bias_m_read_readvariableop?savev2_adam_batch_normalization_160_gamma_m_read_readvariableop>savev2_adam_batch_normalization_160_beta_m_read_readvariableop3savev2_adam_conv2d_170_kernel_m_read_readvariableop1savev2_adam_conv2d_170_bias_m_read_readvariableop?savev2_adam_batch_normalization_161_gamma_m_read_readvariableop>savev2_adam_batch_normalization_161_beta_m_read_readvariableop3savev2_adam_conv2d_171_kernel_m_read_readvariableop1savev2_adam_conv2d_171_bias_m_read_readvariableop?savev2_adam_batch_normalization_162_gamma_m_read_readvariableop>savev2_adam_batch_normalization_162_beta_m_read_readvariableop1savev2_adam_dense_37_kernel_m_read_readvariableop/savev2_adam_dense_37_bias_m_read_readvariableop3savev2_adam_conv2d_167_kernel_v_read_readvariableop1savev2_adam_conv2d_167_bias_v_read_readvariableop?savev2_adam_batch_normalization_158_gamma_v_read_readvariableop>savev2_adam_batch_normalization_158_beta_v_read_readvariableop3savev2_adam_conv2d_168_kernel_v_read_readvariableop1savev2_adam_conv2d_168_bias_v_read_readvariableop?savev2_adam_batch_normalization_159_gamma_v_read_readvariableop>savev2_adam_batch_normalization_159_beta_v_read_readvariableop3savev2_adam_conv2d_169_kernel_v_read_readvariableop1savev2_adam_conv2d_169_bias_v_read_readvariableop?savev2_adam_batch_normalization_160_gamma_v_read_readvariableop>savev2_adam_batch_normalization_160_beta_v_read_readvariableop3savev2_adam_conv2d_170_kernel_v_read_readvariableop1savev2_adam_conv2d_170_bias_v_read_readvariableop?savev2_adam_batch_normalization_161_gamma_v_read_readvariableop>savev2_adam_batch_normalization_161_beta_v_read_readvariableop3savev2_adam_conv2d_171_kernel_v_read_readvariableop1savev2_adam_conv2d_171_bias_v_read_readvariableop?savev2_adam_batch_normalization_162_gamma_v_read_readvariableop>savev2_adam_batch_normalization_162_beta_v_read_readvariableop1savev2_adam_dense_37_kernel_v_read_readvariableop/savev2_adam_dense_37_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *d
dtypesZ
X2V	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ä
_input_shapes²
¯: ::::::::::::: : : : : : : @:@:@:@:@:@:@::::::	M
:
: : : : : : : : : ::::::::: : : : : @:@:@:@:@::::	M
:
::::::::: : : : : @:@:@:@:@::::	M
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::%!

_output_shapes
:	M
:  

_output_shapes
:
:!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :,*(
&
_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
::,2(
&
_output_shapes
: : 3

_output_shapes
: : 4

_output_shapes
: : 5

_output_shapes
: :,6(
&
_output_shapes
: @: 7

_output_shapes
:@: 8

_output_shapes
:@: 9

_output_shapes
:@:-:)
'
_output_shapes
:@:!;

_output_shapes	
::!<

_output_shapes	
::!=

_output_shapes	
::%>!

_output_shapes
:	M
: ?

_output_shapes
:
:,@(
&
_output_shapes
:: A

_output_shapes
:: B

_output_shapes
:: C

_output_shapes
::,D(
&
_output_shapes
:: E

_output_shapes
:: F

_output_shapes
:: G

_output_shapes
::,H(
&
_output_shapes
: : I

_output_shapes
: : J

_output_shapes
: : K

_output_shapes
: :,L(
&
_output_shapes
: @: M

_output_shapes
:@: N

_output_shapes
:@: O

_output_shapes
:@:-P)
'
_output_shapes
:@:!Q

_output_shapes	
::!R

_output_shapes	
::!S

_output_shapes	
::%T!

_output_shapes
:	M
: U

_output_shapes
:
:V

_output_shapes
: 

i
M__inference_max_pooling2d_170_layer_call_and_return_conditional_losses_423910

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_167_layer_call_and_return_conditional_losses_423562

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ °: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
 
_user_specified_nameinputs
¥
G
+__inference_dropout_40_layer_call_fn_424018

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_40_layer_call_and_return_conditional_losses_422431a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿM:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
 
_user_specified_nameinputs
	
Ó
8__inference_batch_normalization_160_layer_call_fn_423772

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_160_layer_call_and_return_conditional_losses_422101
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_167_layer_call_and_return_conditional_losses_423634

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
N
2__inference_max_pooling2d_169_layer_call_fn_423813

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_169_layer_call_and_return_conditional_losses_422121
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã[
¥
I__inference_sequential_21_layer_call_and_return_conditional_losses_422451

inputs+
conv2d_167_422295:
conv2d_167_422297:,
batch_normalization_158_422300:,
batch_normalization_158_422302:,
batch_normalization_158_422304:,
batch_normalization_158_422306:+
conv2d_168_422322:
conv2d_168_422324:,
batch_normalization_159_422327:,
batch_normalization_159_422329:,
batch_normalization_159_422331:,
batch_normalization_159_422333:+
conv2d_169_422349: 
conv2d_169_422351: ,
batch_normalization_160_422354: ,
batch_normalization_160_422356: ,
batch_normalization_160_422358: ,
batch_normalization_160_422360: +
conv2d_170_422376: @
conv2d_170_422378:@,
batch_normalization_161_422381:@,
batch_normalization_161_422383:@,
batch_normalization_161_422385:@,
batch_normalization_161_422387:@,
conv2d_171_422403:@ 
conv2d_171_422405:	-
batch_normalization_162_422408:	-
batch_normalization_162_422410:	-
batch_normalization_162_422412:	-
batch_normalization_162_422414:	"
dense_37_422445:	M

dense_37_422447:

identity¢/batch_normalization_158/StatefulPartitionedCall¢/batch_normalization_159/StatefulPartitionedCall¢/batch_normalization_160/StatefulPartitionedCall¢/batch_normalization_161/StatefulPartitionedCall¢/batch_normalization_162/StatefulPartitionedCall¢"conv2d_167/StatefulPartitionedCall¢"conv2d_168/StatefulPartitionedCall¢"conv2d_169/StatefulPartitionedCall¢"conv2d_170/StatefulPartitionedCall¢"conv2d_171/StatefulPartitionedCall¢ dense_37/StatefulPartitionedCall
"conv2d_167/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_167_422295conv2d_167_422297*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_167_layer_call_and_return_conditional_losses_422294
/batch_normalization_158/StatefulPartitionedCallStatefulPartitionedCall+conv2d_167/StatefulPartitionedCall:output:0batch_normalization_158_422300batch_normalization_158_422302batch_normalization_158_422304batch_normalization_158_422306*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_158_layer_call_and_return_conditional_losses_421918
!max_pooling2d_167/PartitionedCallPartitionedCall8batch_normalization_158/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_167_layer_call_and_return_conditional_losses_421969¦
"conv2d_168/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_167/PartitionedCall:output:0conv2d_168_422322conv2d_168_422324*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_168_layer_call_and_return_conditional_losses_422321
/batch_normalization_159/StatefulPartitionedCallStatefulPartitionedCall+conv2d_168/StatefulPartitionedCall:output:0batch_normalization_159_422327batch_normalization_159_422329batch_normalization_159_422331batch_normalization_159_422333*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_159_layer_call_and_return_conditional_losses_421994
!max_pooling2d_168/PartitionedCallPartitionedCall8batch_normalization_159/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿFj* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_168_layer_call_and_return_conditional_losses_422045¤
"conv2d_169/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_168/PartitionedCall:output:0conv2d_169_422349conv2d_169_422351*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_169_layer_call_and_return_conditional_losses_422348
/batch_normalization_160/StatefulPartitionedCallStatefulPartitionedCall+conv2d_169/StatefulPartitionedCall:output:0batch_normalization_160_422354batch_normalization_160_422356batch_normalization_160_422358batch_normalization_160_422360*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDh *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_160_layer_call_and_return_conditional_losses_422070
!max_pooling2d_169/PartitionedCallPartitionedCall8batch_normalization_160/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"4 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_169_layer_call_and_return_conditional_losses_422121¤
"conv2d_170/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_169/PartitionedCall:output:0conv2d_170_422376conv2d_170_422378*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_170_layer_call_and_return_conditional_losses_422375
/batch_normalization_161/StatefulPartitionedCallStatefulPartitionedCall+conv2d_170/StatefulPartitionedCall:output:0batch_normalization_161_422381batch_normalization_161_422383batch_normalization_161_422385batch_normalization_161_422387*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_161_layer_call_and_return_conditional_losses_422146
!max_pooling2d_170/PartitionedCallPartitionedCall8batch_normalization_161/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_170_layer_call_and_return_conditional_losses_422197¥
"conv2d_171/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_170/PartitionedCall:output:0conv2d_171_422403conv2d_171_422405*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_171_layer_call_and_return_conditional_losses_422402
/batch_normalization_162/StatefulPartitionedCallStatefulPartitionedCall+conv2d_171/StatefulPartitionedCall:output:0batch_normalization_162_422408batch_normalization_162_422410batch_normalization_162_422412batch_normalization_162_422414*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_162_layer_call_and_return_conditional_losses_422222
!max_pooling2d_171/PartitionedCallPartitionedCall8batch_normalization_162/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_171_layer_call_and_return_conditional_losses_422273á
flatten_38/PartitionedCallPartitionedCall*max_pooling2d_171/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_38_layer_call_and_return_conditional_losses_422424Ú
dropout_40/PartitionedCallPartitionedCall#flatten_38/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_40_layer_call_and_return_conditional_losses_422431
 dense_37/StatefulPartitionedCallStatefulPartitionedCall#dropout_40/PartitionedCall:output:0dense_37_422445dense_37_422447*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_422444x
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp0^batch_normalization_158/StatefulPartitionedCall0^batch_normalization_159/StatefulPartitionedCall0^batch_normalization_160/StatefulPartitionedCall0^batch_normalization_161/StatefulPartitionedCall0^batch_normalization_162/StatefulPartitionedCall#^conv2d_167/StatefulPartitionedCall#^conv2d_168/StatefulPartitionedCall#^conv2d_169/StatefulPartitionedCall#^conv2d_170/StatefulPartitionedCall#^conv2d_171/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ °: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_158/StatefulPartitionedCall/batch_normalization_158/StatefulPartitionedCall2b
/batch_normalization_159/StatefulPartitionedCall/batch_normalization_159/StatefulPartitionedCall2b
/batch_normalization_160/StatefulPartitionedCall/batch_normalization_160/StatefulPartitionedCall2b
/batch_normalization_161/StatefulPartitionedCall/batch_normalization_161/StatefulPartitionedCall2b
/batch_normalization_162/StatefulPartitionedCall/batch_normalization_162/StatefulPartitionedCall2H
"conv2d_167/StatefulPartitionedCall"conv2d_167/StatefulPartitionedCall2H
"conv2d_168/StatefulPartitionedCall"conv2d_168/StatefulPartitionedCall2H
"conv2d_169/StatefulPartitionedCall"conv2d_169/StatefulPartitionedCall2H
"conv2d_170/StatefulPartitionedCall"conv2d_170/StatefulPartitionedCall2H
"conv2d_171/StatefulPartitionedCall"conv2d_171/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
 
_user_specified_nameinputs
Ê
b
F__inference_flatten_38_layer_call_and_return_conditional_losses_422424

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿMY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
Æ
S__inference_batch_normalization_162_layer_call_and_return_conditional_losses_422253

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó
á
.__inference_sequential_21_layer_call_fn_423216

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: @

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@%

unknown_23:@

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	M


unknown_30:

identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*8
_read_only_resource_inputs
	
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_21_layer_call_and_return_conditional_losses_422764o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ °: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
 
_user_specified_nameinputs
î
 
+__inference_conv2d_170_layer_call_fn_423827

inputs!
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_170_layer_call_and_return_conditional_losses_422375w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ"4 : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"4 
 
_user_specified_nameinputs
Ü
Â
S__inference_batch_normalization_160_layer_call_and_return_conditional_losses_423808

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_167_layer_call_and_return_conditional_losses_421969

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
d
F__inference_dropout_40_layer_call_and_return_conditional_losses_424028

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿM:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ç
serving_default³
W
conv2d_167_inputC
"serving_default_conv2d_167_input:0ÿÿÿÿÿÿÿÿÿ °<
dense_370
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:¦×
÷
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer-15
layer-16
layer_with_weights-10
layer-17
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
»

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
$axis
	%gamma
&beta
'moving_mean
(moving_variance
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
»

5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
=axis
	>gamma
?beta
@moving_mean
Amoving_variance
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Nkernel
Obias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
Vaxis
	Wgamma
Xbeta
Ymoving_mean
Zmoving_variance
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
»

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
	variables
 trainable_variables
¡regularization_losses
¢	keras_api
£_random_generator
¤__call__
+¥&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
¦kernel
	§bias
¨	variables
©trainable_variables
ªregularization_losses
«	keras_api
¬__call__
+­&call_and_return_all_conditional_losses"
_tf_keras_layer

	®iter
¯beta_1
°beta_2

±decay
²learning_ratemm%m &m¡5m¢6m£>m¤?m¥Nm¦Om§Wm¨Xm©gmªhm«pm¬qm­	m®	m¯	m°	m±	¦m²	§m³v´vµ%v¶&v·5v¸6v¹>vº?v»Nv¼Ov½Wv¾Xv¿gvÀhvÁpvÂqvÃ	vÄ	vÅ	vÆ	vÇ	¦vÈ	§vÉ"
	optimizer

0
1
%2
&3
'4
(5
56
67
>8
?9
@10
A11
N12
O13
W14
X15
Y16
Z17
g18
h19
p20
q21
r22
s23
24
25
26
27
28
29
¦30
§31"
trackable_list_wrapper
Ì
0
1
%2
&3
54
65
>6
?7
N8
O9
W10
X11
g12
h13
p14
q15
16
17
18
19
¦20
§21"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
.__inference_sequential_21_layer_call_fn_422518
.__inference_sequential_21_layer_call_fn_423147
.__inference_sequential_21_layer_call_fn_423216
.__inference_sequential_21_layer_call_fn_422900À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
I__inference_sequential_21_layer_call_and_return_conditional_losses_423340
I__inference_sequential_21_layer_call_and_return_conditional_losses_423471
I__inference_sequential_21_layer_call_and_return_conditional_losses_422986
I__inference_sequential_21_layer_call_and_return_conditional_losses_423072À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÕBÒ
!__inference__wrapped_model_421896conv2d_167_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
-
¸serving_default"
signature_map
+:)2conv2d_167/kernel
:2conv2d_167/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_conv2d_167_layer_call_fn_423551¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_167_layer_call_and_return_conditional_losses_423562¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
+:)2batch_normalization_158/gamma
*:(2batch_normalization_158/beta
3:1 (2#batch_normalization_158/moving_mean
7:5 (2'batch_normalization_158/moving_variance
<
%0
&1
'2
(3"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¾non_trainable_variables
¿layers
Àmetrics
 Álayer_regularization_losses
Âlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_158_layer_call_fn_423575
8__inference_batch_normalization_158_layer_call_fn_423588´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
S__inference_batch_normalization_158_layer_call_and_return_conditional_losses_423606
S__inference_batch_normalization_158_layer_call_and_return_conditional_losses_423624´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
Ü2Ù
2__inference_max_pooling2d_167_layer_call_fn_423629¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_max_pooling2d_167_layer_call_and_return_conditional_losses_423634¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
+:)2conv2d_168/kernel
:2conv2d_168/bias
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_conv2d_168_layer_call_fn_423643¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_168_layer_call_and_return_conditional_losses_423654¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
+:)2batch_normalization_159/gamma
*:(2batch_normalization_159/beta
3:1 (2#batch_normalization_159/moving_mean
7:5 (2'batch_normalization_159/moving_variance
<
>0
?1
@2
A3"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_159_layer_call_fn_423667
8__inference_batch_normalization_159_layer_call_fn_423680´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
S__inference_batch_normalization_159_layer_call_and_return_conditional_losses_423698
S__inference_batch_normalization_159_layer_call_and_return_conditional_losses_423716´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
Ü2Ù
2__inference_max_pooling2d_168_layer_call_fn_423721¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_max_pooling2d_168_layer_call_and_return_conditional_losses_423726¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
+:) 2conv2d_169/kernel
: 2conv2d_169/bias
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_conv2d_169_layer_call_fn_423735¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_169_layer_call_and_return_conditional_losses_423746¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
+:) 2batch_normalization_160/gamma
*:( 2batch_normalization_160/beta
3:1  (2#batch_normalization_160/moving_mean
7:5  (2'batch_normalization_160/moving_variance
<
W0
X1
Y2
Z3"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ünon_trainable_variables
Ýlayers
Þmetrics
 ßlayer_regularization_losses
àlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_160_layer_call_fn_423759
8__inference_batch_normalization_160_layer_call_fn_423772´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
S__inference_batch_normalization_160_layer_call_and_return_conditional_losses_423790
S__inference_batch_normalization_160_layer_call_and_return_conditional_losses_423808´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
Ü2Ù
2__inference_max_pooling2d_169_layer_call_fn_423813¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_max_pooling2d_169_layer_call_and_return_conditional_losses_423818¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
+:) @2conv2d_170/kernel
:@2conv2d_170/bias
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_conv2d_170_layer_call_fn_423827¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_170_layer_call_and_return_conditional_losses_423838¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
+:)@2batch_normalization_161/gamma
*:(@2batch_normalization_161/beta
3:1@ (2#batch_normalization_161/moving_mean
7:5@ (2'batch_normalization_161/moving_variance
<
p0
q1
r2
s3"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_161_layer_call_fn_423851
8__inference_batch_normalization_161_layer_call_fn_423864´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
S__inference_batch_normalization_161_layer_call_and_return_conditional_losses_423882
S__inference_batch_normalization_161_layer_call_and_return_conditional_losses_423900´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ü2Ù
2__inference_max_pooling2d_170_layer_call_fn_423905¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_max_pooling2d_170_layer_call_and_return_conditional_losses_423910¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,:*@2conv2d_171/kernel
:2conv2d_171/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_conv2d_171_layer_call_fn_423919¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_171_layer_call_and_return_conditional_losses_423930¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
,:*2batch_normalization_162/gamma
+:)2batch_normalization_162/beta
4:2 (2#batch_normalization_162/moving_mean
8:6 (2'batch_normalization_162/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_162_layer_call_fn_423943
8__inference_batch_normalization_162_layer_call_fn_423956´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
S__inference_batch_normalization_162_layer_call_and_return_conditional_losses_423974
S__inference_batch_normalization_162_layer_call_and_return_conditional_losses_423992´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ü2Ù
2__inference_max_pooling2d_171_layer_call_fn_423997¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_max_pooling2d_171_layer_call_and_return_conditional_losses_424002¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_flatten_38_layer_call_fn_424007¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_flatten_38_layer_call_and_return_conditional_losses_424013¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
 trainable_variables
¡regularization_losses
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_dropout_40_layer_call_fn_424018
+__inference_dropout_40_layer_call_fn_424023´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_40_layer_call_and_return_conditional_losses_424028
F__inference_dropout_40_layer_call_and_return_conditional_losses_424040´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
": 	M
2dense_37/kernel
:
2dense_37/bias
0
¦0
§1"
trackable_list_wrapper
0
¦0
§1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¨	variables
©trainable_variables
ªregularization_losses
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_37_layer_call_fn_424049¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_37_layer_call_and_return_conditional_losses_424060¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
h
'0
(1
@2
A3
Y4
Z5
r6
s7
8
9"
trackable_list_wrapper
¦
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÔBÑ
$__inference_signature_wrapper_423542conv2d_167_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
.
'0
(1"
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
.
@0
A1"
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
.
Y0
Z1"
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
.
r0
s1"
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
0
0
1"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

total

count
	variables
	keras_api"
_tf_keras_metric
c

total

count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
0:.2Adam/conv2d_167/kernel/m
": 2Adam/conv2d_167/bias/m
0:.2$Adam/batch_normalization_158/gamma/m
/:-2#Adam/batch_normalization_158/beta/m
0:.2Adam/conv2d_168/kernel/m
": 2Adam/conv2d_168/bias/m
0:.2$Adam/batch_normalization_159/gamma/m
/:-2#Adam/batch_normalization_159/beta/m
0:. 2Adam/conv2d_169/kernel/m
":  2Adam/conv2d_169/bias/m
0:. 2$Adam/batch_normalization_160/gamma/m
/:- 2#Adam/batch_normalization_160/beta/m
0:. @2Adam/conv2d_170/kernel/m
": @2Adam/conv2d_170/bias/m
0:.@2$Adam/batch_normalization_161/gamma/m
/:-@2#Adam/batch_normalization_161/beta/m
1:/@2Adam/conv2d_171/kernel/m
#:!2Adam/conv2d_171/bias/m
1:/2$Adam/batch_normalization_162/gamma/m
0:.2#Adam/batch_normalization_162/beta/m
':%	M
2Adam/dense_37/kernel/m
 :
2Adam/dense_37/bias/m
0:.2Adam/conv2d_167/kernel/v
": 2Adam/conv2d_167/bias/v
0:.2$Adam/batch_normalization_158/gamma/v
/:-2#Adam/batch_normalization_158/beta/v
0:.2Adam/conv2d_168/kernel/v
": 2Adam/conv2d_168/bias/v
0:.2$Adam/batch_normalization_159/gamma/v
/:-2#Adam/batch_normalization_159/beta/v
0:. 2Adam/conv2d_169/kernel/v
":  2Adam/conv2d_169/bias/v
0:. 2$Adam/batch_normalization_160/gamma/v
/:- 2#Adam/batch_normalization_160/beta/v
0:. @2Adam/conv2d_170/kernel/v
": @2Adam/conv2d_170/bias/v
0:.@2$Adam/batch_normalization_161/gamma/v
/:-@2#Adam/batch_normalization_161/beta/v
1:/@2Adam/conv2d_171/kernel/v
#:!2Adam/conv2d_171/bias/v
1:/2$Adam/batch_normalization_162/gamma/v
0:.2#Adam/batch_normalization_162/beta/v
':%	M
2Adam/dense_37/kernel/v
 :
2Adam/dense_37/bias/vÊ
!__inference__wrapped_model_421896¤(%&'(56>?@ANOWXYZghpqrs¦§C¢@
9¢6
41
conv2d_167_inputÿÿÿÿÿÿÿÿÿ °
ª "3ª0
.
dense_37"
dense_37ÿÿÿÿÿÿÿÿÿ
î
S__inference_batch_normalization_158_layer_call_and_return_conditional_losses_423606%&'(M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 î
S__inference_batch_normalization_158_layer_call_and_return_conditional_losses_423624%&'(M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
8__inference_batch_normalization_158_layer_call_fn_423575%&'(M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
8__inference_batch_normalization_158_layer_call_fn_423588%&'(M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
S__inference_batch_normalization_159_layer_call_and_return_conditional_losses_423698>?@AM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 î
S__inference_batch_normalization_159_layer_call_and_return_conditional_losses_423716>?@AM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
8__inference_batch_normalization_159_layer_call_fn_423667>?@AM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
8__inference_batch_normalization_159_layer_call_fn_423680>?@AM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
S__inference_batch_normalization_160_layer_call_and_return_conditional_losses_423790WXYZM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 î
S__inference_batch_normalization_160_layer_call_and_return_conditional_losses_423808WXYZM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Æ
8__inference_batch_normalization_160_layer_call_fn_423759WXYZM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Æ
8__inference_batch_normalization_160_layer_call_fn_423772WXYZM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ î
S__inference_batch_normalization_161_layer_call_and_return_conditional_losses_423882pqrsM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 î
S__inference_batch_normalization_161_layer_call_and_return_conditional_losses_423900pqrsM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Æ
8__inference_batch_normalization_161_layer_call_fn_423851pqrsM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Æ
8__inference_batch_normalization_161_layer_call_fn_423864pqrsM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ô
S__inference_batch_normalization_162_layer_call_and_return_conditional_losses_423974N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ô
S__inference_batch_normalization_162_layer_call_and_return_conditional_losses_423992N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ì
8__inference_batch_normalization_162_layer_call_fn_423943N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÌ
8__inference_batch_normalization_162_layer_call_fn_423956N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
F__inference_conv2d_167_layer_call_and_return_conditional_losses_423562p9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ °
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ®
 
+__inference_conv2d_167_layer_call_fn_423551c9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ °
ª ""ÿÿÿÿÿÿÿÿÿ®º
F__inference_conv2d_168_layer_call_and_return_conditional_losses_423654p569¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ×
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿÕ
 
+__inference_conv2d_168_layer_call_fn_423643c569¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ×
ª ""ÿÿÿÿÿÿÿÿÿÕ¶
F__inference_conv2d_169_layer_call_and_return_conditional_losses_423746lNO7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿFj
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿDh 
 
+__inference_conv2d_169_layer_call_fn_423735_NO7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿFj
ª " ÿÿÿÿÿÿÿÿÿDh ¶
F__inference_conv2d_170_layer_call_and_return_conditional_losses_423838lgh7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ"4 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 2@
 
+__inference_conv2d_170_layer_call_fn_423827_gh7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ"4 
ª " ÿÿÿÿÿÿÿÿÿ 2@¹
F__inference_conv2d_171_layer_call_and_return_conditional_losses_423930o7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_conv2d_171_layer_call_fn_423919b7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "!ÿÿÿÿÿÿÿÿÿ§
D__inference_dense_37_layer_call_and_return_conditional_losses_424060_¦§0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿM
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
)__inference_dense_37_layer_call_fn_424049R¦§0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿM
ª "ÿÿÿÿÿÿÿÿÿ
¨
F__inference_dropout_40_layer_call_and_return_conditional_losses_424028^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿM
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿM
 ¨
F__inference_dropout_40_layer_call_and_return_conditional_losses_424040^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿM
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿM
 
+__inference_dropout_40_layer_call_fn_424018Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿM
p 
ª "ÿÿÿÿÿÿÿÿÿM
+__inference_dropout_40_layer_call_fn_424023Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿM
p
ª "ÿÿÿÿÿÿÿÿÿM¬
F__inference_flatten_38_layer_call_and_return_conditional_losses_424013b8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿM
 
+__inference_flatten_38_layer_call_fn_424007U8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿMð
M__inference_max_pooling2d_167_layer_call_and_return_conditional_losses_423634R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_167_layer_call_fn_423629R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_168_layer_call_and_return_conditional_losses_423726R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_168_layer_call_fn_423721R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_169_layer_call_and_return_conditional_losses_423818R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_169_layer_call_fn_423813R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_170_layer_call_and_return_conditional_losses_423910R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_170_layer_call_fn_423905R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_171_layer_call_and_return_conditional_losses_424002R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_171_layer_call_fn_423997R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_sequential_21_layer_call_and_return_conditional_losses_422986(%&'(56>?@ANOWXYZghpqrs¦§K¢H
A¢>
41
conv2d_167_inputÿÿÿÿÿÿÿÿÿ °
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ì
I__inference_sequential_21_layer_call_and_return_conditional_losses_423072(%&'(56>?@ANOWXYZghpqrs¦§K¢H
A¢>
41
conv2d_167_inputÿÿÿÿÿÿÿÿÿ °
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 â
I__inference_sequential_21_layer_call_and_return_conditional_losses_423340(%&'(56>?@ANOWXYZghpqrs¦§A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ °
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 â
I__inference_sequential_21_layer_call_and_return_conditional_losses_423471(%&'(56>?@ANOWXYZghpqrs¦§A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ °
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ä
.__inference_sequential_21_layer_call_fn_422518(%&'(56>?@ANOWXYZghpqrs¦§K¢H
A¢>
41
conv2d_167_inputÿÿÿÿÿÿÿÿÿ °
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
Ä
.__inference_sequential_21_layer_call_fn_422900(%&'(56>?@ANOWXYZghpqrs¦§K¢H
A¢>
41
conv2d_167_inputÿÿÿÿÿÿÿÿÿ °
p

 
ª "ÿÿÿÿÿÿÿÿÿ
º
.__inference_sequential_21_layer_call_fn_423147(%&'(56>?@ANOWXYZghpqrs¦§A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ °
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
º
.__inference_sequential_21_layer_call_fn_423216(%&'(56>?@ANOWXYZghpqrs¦§A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ °
p

 
ª "ÿÿÿÿÿÿÿÿÿ
á
$__inference_signature_wrapper_423542¸(%&'(56>?@ANOWXYZghpqrs¦§W¢T
¢ 
MªJ
H
conv2d_167_input41
conv2d_167_inputÿÿÿÿÿÿÿÿÿ °"3ª0
.
dense_37"
dense_37ÿÿÿÿÿÿÿÿÿ
