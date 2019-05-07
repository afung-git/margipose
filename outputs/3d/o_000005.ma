createNode transform -s -n "root_o_000005";
	rename -uid "2e648627-d5d5-453d-9184-a3469e9baa80";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_o_000005";
	rename -uid "6eb91176-cf80-4880-9b09-4248390c93d9";
	setAttr ".t" -type "double3" 16.660360991954803 -3.372238576412201 -0.07931888103485107 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "f8e9cee8-97c9-425b-99f9-d6b66f333514";
	setAttr ".t" -type "double3" -11.093499884009361 0.33408403396606445 0.3401963971555233 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "8c9263b8-8a03-4a19-a335-eb880561bd43";
	setAttr ".t" -type "double3" -1.595650240778923 39.96201604604721 -19.2546128295362 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "2fd83806-d2b8-42f8-9930-e5f55c731606";
	setAttr ".t" -type "double3" 4.901983588933945 47.36528396606445 10.358374565839767 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "5874a2f9-2d5f-4957-98f2-0d9d5045e0b0";
	setAttr ".t" -type "double3" 10.929007828235626 -0.3124542534351349 -0.692336168140173 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "2031eb1b-85f9-438d-a457-a8028eb6b284";
	setAttr ".t" -type "double3" 3.2248765230178833 40.0811068713665 9.632123913615942 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "af754d51-cd1f-424f-9858-6c42ca7e3ec7";
	setAttr ".t" -type "double3" 1.0946542024612427 43.66886019706726 3.63808274269104 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "46995c00-632c-4aef-b79a-d9aa0b5ef122";
	setAttr ".t" -type "double3" -0.5807816982269287 -32.61941522359848 -1.267408300191164 ;
createNode joint -n "neck" -p "spine";
	rename -uid "ddc70f9c-f7f2-4d63-8ac8-078e554ab21c";
	setAttr ".t" -type "double3" -1.4456570148468018 -41.042742133140564 -2.378712873905897 ;
createNode joint -n "head" -p "neck";
	rename -uid "0e9c3a18-d2dc-459e-8ffa-a085d87de5c2";
	setAttr ".t" -type "double3" 0.09025633335113525 -14.406043291091919 -0.5553837865591049 ;
createNode joint -n "head_top" -p "head";
	rename -uid "d84d3604-fc08-47ab-94dc-205bf17f805c";
	setAttr ".t" -type "double3" -0.30733048915863037 -3.5962820053100586 14.682888612151146 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "f2886ae9-3c58-49dc-97f0-c93161fcf04f";
	setAttr ".t" -type "double3" -22.18964472413063 9.347641468048096 -1.5078868716955185 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "2e658165-8be0-4208-8cfd-892dcbe2b621";
	setAttr ".t" -type "double3" -3.9083607494831085 27.861881256103516 -16.48183725774288 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "fa83be78-8f50-41d8-a8fa-55e2778be564";
	setAttr ".t" -type "double3" 5.9273067861795425 20.497559010982513 -19.829517602920532 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "b689d2e6-bf0a-4636-86eb-443badb7628d";
	setAttr ".t" -type "double3" 22.018416225910187 9.666448831558228 -3.7007130682468414 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "c0f2b216-04a3-4b9a-b1bc-e038f7df53fd";
	setAttr ".t" -type "double3" 8.513486385345459 29.56503927707672 -19.145314395427704 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "be75d0b1-d023-4b44-8281-b2cdc73875b6";
	setAttr ".t" -type "double3" -3.8582831621170044 20.379836857318878 -18.34453046321869 ;
