createNode transform -s -n "root_c_000004";
	rename -uid "e957b13b-ba19-4b06-92d5-181f01e3a0a0";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_c_000004";
	rename -uid "27f0c8bd-f013-49b7-8167-12e5bfd4536f";
	setAttr ".t" -type "double3" -7.582373172044754 45.894694328308105 0.3806969150900841 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "b4109282-f234-4328-a3de-d34548eeb0fc";
	setAttr ".t" -type "double3" -28.601276129484177 6.196242570877075 -1.366482861340046 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "0ecb94b8-978b-4d0d-88dc-94f56552b0b8";
	setAttr ".t" -type "double3" 17.134569585323334 6.981760263442993 4.782116040587425 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "c6865f0b-c76b-4615-bb0c-31b7e3726356";
	setAttr ".t" -type "double3" 17.60159432888031 8.846408128738403 61.89660727977753 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "89b7a1a7-9b16-4074-b65a-09bc224928d4";
	setAttr ".t" -type "double3" 15.61378687620163 -2.590310573577881 4.205629043281078 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "74895ad1-99a4-4774-a6c5-c7d258edab97";
	setAttr ".t" -type "double3" 41.012924164533615 -21.2281733751297 -5.092879012227058 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "c009317d-5f9f-494a-a62f-3189f0b01f17";
	setAttr ".t" -type "double3" -34.631578624248505 43.317022919654846 71.54487669467926 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "6ad147e6-12a1-4961-a3ef-3e10cb60c600";
	setAttr ".t" -type "double3" -1.3179890811443329 -42.186445742845535 -27.15264167636633 ;
createNode joint -n "neck" -p "spine";
	rename -uid "3d535670-3ddf-460e-94eb-fc9468f6f2f1";
	setAttr ".t" -type "double3" 0.9950779378414154 -44.07956227660179 -18.08462142944336 ;
createNode joint -n "head" -p "neck";
	rename -uid "25c3e205-a8b5-417b-bea2-517abb60eb21";
	setAttr ".t" -type "double3" 1.0043740272521973 -19.297006726264954 -1.0041594505310059 ;
createNode joint -n "head_top" -p "head";
	rename -uid "2193acf5-ba6b-4224-9c22-e4eaecd393c4";
	setAttr ".t" -type "double3" 1.3947013765573502 -20.150798559188843 21.1289644241333 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "56aff0d4-3a0e-4376-92d9-b62384649df7";
	setAttr ".t" -type "double3" -31.626703590154648 3.075742721557617 -6.470984220504761 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "07e6ea62-5452-4fed-acb3-3e467d6679bd";
	setAttr ".t" -type "double3" -28.30897569656372 4.658737778663635 -13.588547706604004 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "61fb58b5-0bfa-4cbc-afbe-0d0d884b6199";
	setAttr ".t" -type "double3" -19.927704334259033 1.4904201030731201 -26.979458332061768 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "c8d889c6-67dc-4af9-ab9a-8e8d344133d3";
	setAttr ".t" -type "double3" 29.593989998102188 10.821974277496338 13.900399208068848 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "b60b0a9f-2fe2-46f7-9aaf-bcfad163e043";
	setAttr ".t" -type "double3" 39.412569999694824 15.667089819908142 30.75938131660223 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "7323f063-a4b9-4aaf-bce1-73b394e32877";
	setAttr ".t" -type "double3" 29.029935598373413 4.5751869678497314 14.06367663294077 ;
