createNode transform -s -n "root_e_000005";
	rename -uid "c7de133a-64f3-44d1-b4f9-87b882fdccb7";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_e_000005";
	rename -uid "ded70f5e-846a-4169-9d40-69edd1730263";
	setAttr ".t" -type "double3" 5.563322827219963 16.41838699579239 0.13442561030387878 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "e1d767ca-09bf-4919-8cc4-02034d1e2f93";
	setAttr ".t" -type "double3" -9.44230630993843 -0.4162326455116272 1.8775716423988342 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "0885bd7b-c018-429b-821f-124241a34a40";
	setAttr ".t" -type "double3" -12.833300605416298 32.219791412353516 -34.65970233082771 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "2003bd5d-95c0-4835-807d-8a4cc5c46a67";
	setAttr ".t" -type "double3" 21.913830190896988 9.713345766067505 35.0098030641675 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "2afeef45-abe1-46fb-b1b8-d6386a2c25d6";
	setAttr ".t" -type "double3" 9.612684324383736 0.2648472785949707 -1.3897402212023735 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "6617d299-c5ec-479a-9391-72c482cd2a15";
	setAttr ".t" -type "double3" 13.96074891090393 29.335735738277435 -39.21899888664484 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "b8a5490b-4114-4d09-b05f-992bf64b01d3";
	setAttr ".t" -type "double3" -22.06125482916832 6.384903192520142 43.245729804039 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "6c8d831d-1ed2-4438-8ed2-e46abe0003d2";
	setAttr ".t" -type "double3" 0.737546756863594 -25.15052855014801 -2.2503338754177094 ;
createNode joint -n "neck" -p "spine";
	rename -uid "16eb20fb-76fa-4f32-bf20-84dc5c142dcc";
	setAttr ".t" -type "double3" -0.04698038101196289 -28.99293452501297 -10.892325639724731 ;
createNode joint -n "head" -p "neck";
	rename -uid "9afd7907-be06-483a-8125-58029764d166";
	setAttr ".t" -type "double3" -0.8632153272628784 -12.97437846660614 -4.543134570121765 ;
createNode joint -n "head_top" -p "head";
	rename -uid "a5880097-b882-4466-9b18-ccd477ed29f0";
	setAttr ".t" -type "double3" -0.6107181310653687 -13.357019424438477 -0.720822811126709 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "4095b47a-24f4-487b-bcf1-f9ddbbd312db";
	setAttr ".t" -type "double3" -16.445069015026093 7.467091083526611 3.2479405403137207 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "d69f9f81-1af7-4e72-b757-1cb1c9d71701";
	setAttr ".t" -type "double3" -13.339397311210632 28.34841199219227 1.2156561017036438 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "90180ba6-9426-43b9-8899-1835c952ca8b";
	setAttr ".t" -type "double3" 6.529375910758972 -8.622337505221367 -24.380449950695038 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "2063be3a-1a70-4c01-8377-d69e896c39a3";
	setAttr ".t" -type "double3" 17.183244228363037 6.291022896766663 -2.542349696159363 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "7b10ff89-6d62-4db4-a836-c681696487f6";
	setAttr ".t" -type "double3" 17.02348291873932 20.629247277975082 -12.986749410629272 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "834267b5-d2fa-42e4-ab6d-95f14084da00";
	setAttr ".t" -type "double3" -24.64260905981064 4.294145852327347 -8.469611406326294 ;
