createNode transform -s -n "root_m_000010";
	rename -uid "613f9a9f-ab5b-4c47-ab16-b5e424e97953";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_m_000010";
	rename -uid "85d5d4d3-c406-4472-aa53-6454e621bd4a";
	setAttr ".t" -type "double3" -4.344738274812698 4.73673976957798 -0.045872293412685394 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "240fd6a9-fcbe-41eb-b9a4-c7f69feffe6e";
	setAttr ".t" -type "double3" -8.048822730779648 2.792978659272194 -4.205804504454136 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "c8151c76-badd-42ae-a219-3dedec155900";
	setAttr ".t" -type "double3" 4.409012943506241 38.08777257800102 2.5510964915156364 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "8ddfa295-9f1b-4426-b9ec-fd59d67aed8b";
	setAttr ".t" -type "double3" -1.3948164880275726 32.00055956840515 11.283284611999989 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "37a8b821-d8be-44f4-a9da-4822ef77f28e";
	setAttr ".t" -type "double3" 8.153632283210754 -2.876100316643715 3.7261607125401497 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "d935beb8-9971-4118-a7d8-182c94715dba";
	setAttr ".t" -type "double3" 18.665006011724472 29.831165075302124 -18.99719499051571 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "6c2eef17-35f8-4a7a-a162-dfb5bbfcf413";
	setAttr ".t" -type "double3" 15.765194594860077 35.70578992366791 -7.9994797706604 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "7391450d-98da-4831-85b2-abe82108c79a";
	setAttr ".t" -type "double3" -7.888691872358322 -21.269363537430763 -4.315769113600254 ;
createNode joint -n "neck" -p "spine";
	rename -uid "8113ecce-3565-46ce-b847-e7d033f989d8";
	setAttr ".t" -type "double3" -7.9299211502075195 -18.727031350135803 -13.49688172340393 ;
createNode joint -n "head" -p "neck";
	rename -uid "2d71b85c-7216-4ce4-9822-59d7737d2dd5";
	setAttr ".t" -type "double3" -4.0985047817230225 -12.336874008178711 -6.670457124710083 ;
createNode joint -n "head_top" -p "head";
	rename -uid "6d1ac839-d834-4f9f-b919-a1b31c0dd811";
	setAttr ".t" -type "double3" -4.357010126113892 -12.78541386127472 -7.156252861022949 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "08a0f6eb-d51a-4199-b639-74cfdb6cf722";
	setAttr ".t" -type "double3" -13.1279855966568 12.419793009757996 -2.6389747858047485 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "fa845166-0126-451d-ad5e-ebdf107dd162";
	setAttr ".t" -type "double3" -2.5629431009292603 26.68931558728218 -1.6689226031303406 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "eef80358-353a-448d-ae3e-d0646d35c2ad";
	setAttr ".t" -type "double3" -5.708405375480652 22.444530576467514 -5.0771549344062805 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "de80932c-1c41-4bf6-ba61-7def842e2085";
	setAttr ".t" -type "double3" 12.595953047275543 -10.8957439661026 6.7610204219818115 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "3ccf2c40-9cc9-48e5-941e-4ed0ba88f7bc";
	setAttr ".t" -type "double3" 16.809995472431183 -10.699236392974854 1.3063505291938782 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "87936d8a-470d-457c-b459-ec9990d2ee0c";
	setAttr ".t" -type "double3" 13.763697445392609 -19.58135962486267 -5.9407442808151245 ;
