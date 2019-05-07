createNode transform -s -n "root_o_000004";
	rename -uid "8a8d5218-5364-40b5-b019-d4bf35bf4fc2";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_o_000004";
	rename -uid "ba5c3b2d-10cd-40e0-a429-c708abffdb7e";
	setAttr ".t" -type "double3" 6.386338174343109 -2.243843860924244 -0.4588955081999302 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "968aa0b4-f995-4eff-9865-d0c2c6a603f0";
	setAttr ".t" -type "double3" -5.723973363637924 1.4989515766501427 -1.6599767841398716 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "13fdea8e-04e8-492f-8591-261d1590444c";
	setAttr ".t" -type "double3" 4.709966108202934 12.562194094061852 46.26306761056185 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "9541c85f-1f32-4ef3-9354-3a7e5f999ee4";
	setAttr ".t" -type "double3" 6.9126565009355545 34.99409705400467 9.712213277816772 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "711986df-0745-4c2a-8c46-ea2c57ac7323";
	setAttr ".t" -type "double3" 3.3190153539180756 -0.9907936677336693 -1.4613059349358082 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "688f5006-a495-483c-94c2-90a1b4762006";
	setAttr ".t" -type "double3" -1.5364192426204681 12.856991589069366 50.370766781270504 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "1fae8c0f-bd49-48fc-b383-a3748b19962b";
	setAttr ".t" -type "double3" 3.5485178232192993 34.485020488500595 31.822049617767334 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "253e651a-7938-4ae0-831a-3f18cba8689e";
	setAttr ".t" -type "double3" -4.119260236620903 -9.389102272689342 -3.4906993620097637 ;
createNode joint -n "neck" -p "spine";
	rename -uid "9273c245-f170-4ea4-9122-461ef4d80481";
	setAttr ".t" -type "double3" -5.388830974698067 -8.437705039978027 13.995938375592232 ;
createNode joint -n "head" -p "neck";
	rename -uid "4a833cb1-2864-4769-ab0a-8da1d35e1e45";
	setAttr ".t" -type "double3" -3.0856765806674957 -7.345357537269592 8.899624645709991 ;
createNode joint -n "head_top" -p "head";
	rename -uid "264dfc3e-f627-4c85-bacf-74263e1d65c9";
	setAttr ".t" -type "double3" -6.666866689920425 -10.396677255630493 -31.511977314949036 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "091115f1-8afa-4d3b-a604-c38c493b4cb2";
	setAttr ".t" -type "double3" -5.277244001626968 2.1188750863075256 -3.4809142351150513 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "0a0698a6-b7f9-4c70-90b6-bd44c10677bd";
	setAttr ".t" -type "double3" 2.155790850520134 10.110725462436676 10.334481298923492 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "78ee2d35-aab4-4da6-8856-1afc23c0c8fd";
	setAttr ".t" -type "double3" -7.780775800347328 -1.6515359282493591 -43.031349778175354 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "b728e266-5ebc-4152-b7d6-8ae0cf6ad59c";
	setAttr ".t" -type "double3" 2.9495246708393097 -2.0564794540405273 -0.9464778006076813 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "fa0087b4-4021-4e39-b327-65f2912e52cb";
	setAttr ".t" -type "double3" -0.39677582681179047 12.488168478012085 7.673702389001846 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "9c39d99c-6283-40d8-a28c-78f2bbb01921";
	setAttr ".t" -type "double3" -4.080531373620033 0.4745468497276306 -42.594070732593536 ;