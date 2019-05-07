createNode transform -s -n "root_000002";
	rename -uid "c92ffba9-b6bb-4076-84be-888a791625d6";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_000002";
	rename -uid "7c5e628f-03ad-4f66-b7fc-bcf64d5ad02d";
	setAttr ".t" -type "double3" -5.665531754493713 -3.4824389964342117 -0.20223306491971016 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "60d91d70-4759-434b-a303-938034a56048";
	setAttr ".t" -type "double3" -9.694965183734894 -1.2091819196939468 -0.47108959406614304 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "a7c2d7d9-8387-47c5-a580-28649440dd74";
	setAttr ".t" -type "double3" -0.4929050803184509 38.33077773451805 -23.657764215022326 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "c171cfb2-db1f-4f89-85a0-89404155805e";
	setAttr ".t" -type "double3" 0.7379278540611267 39.435261487960815 33.86736959218979 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "b80de7f8-698b-4dfd-942f-bb3c2fb78c5d";
	setAttr ".t" -type "double3" 9.458546712994576 0.8073015138506889 1.7763997428119183 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "4978ec0f-0033-4897-9ddc-5fb7211b358f";
	setAttr ".t" -type "double3" -1.172506995499134 40.868464671075344 27.98104677349329 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "8d58688e-aa1c-43ef-8719-6136e87f82ce";
	setAttr ".t" -type "double3" -9.0670021250844 34.6513032913208 17.2893226146698 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "8cc60587-3797-4b8e-a06a-56827e5035f7";
	setAttr ".t" -type "double3" 0.9232137352228165 -29.648391529917717 -3.1049552373588085 ;
createNode joint -n "neck" -p "spine";
	rename -uid "8ba82ae9-17c3-4d77-a6fd-bcb11adfb230";
	setAttr ".t" -type "double3" 0.9791050106287003 -32.57970213890076 -5.854403227567673 ;
createNode joint -n "head" -p "neck";
	rename -uid "133d2f9a-1cb7-4858-b574-88ad4b7ebb81";
	setAttr ".t" -type "double3" -0.20745769143104553 -12.959587574005127 -2.9092542827129364 ;
createNode joint -n "head_top" -p "head";
	rename -uid "91823317-d84f-44ac-a0fb-388793a2550e";
	setAttr ".t" -type "double3" -0.25285184383392334 -12.604910135269165 -3.6349691450595856 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "7173d208-22f3-40e4-8062-1d50859489d4";
	setAttr ".t" -type "double3" -18.526195734739304 5.984383821487427 3.5336345434188843 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "aeeb1860-5831-4a20-afe3-5d8ac86a0eac";
	setAttr ".t" -type "double3" -7.120971381664276 17.411398887634277 14.343398809432983 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "1e333e6e-3f09-4db4-a5fd-e85ed46b0e8f";
	setAttr ".t" -type "double3" -0.7911026477813721 18.484844267368317 -15.151714533567429 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "5a94c7e2-c7fc-44f5-be74-3b22ad13c90f";
	setAttr ".t" -type "double3" 18.361104279756546 6.2246620655059814 -2.2561967372894287 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "fe690f22-0082-4246-a3f1-16fcd8cffbae";
	setAttr ".t" -type "double3" 15.124277770519257 22.818756103515625 -8.072558045387268 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "441b670a-0970-49f8-919e-0b636cfb3763";
	setAttr ".t" -type "double3" -10.43517291545868 2.3932307958602905 -19.896546006202698 ;
