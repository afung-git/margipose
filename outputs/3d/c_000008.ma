createNode transform -s -n "root_c_000008";
	rename -uid "0a1085dd-ff08-462c-ad8f-10588b8609dc";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_c_000008";
	rename -uid "518a8be6-a06d-46f9-994d-5a2bcad776ed";
	setAttr ".t" -type "double3" -39.11636471748352 3.915490210056305 -1.0047821328043938 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "09c0a7e0-771a-4caf-8542-d4f7d2c7a739";
	setAttr ".t" -type "double3" 1.873651146888733 -0.38461051881313324 8.151195757091045 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "d14f9bb1-4d66-4081-8548-979ab9ba2dc0";
	setAttr ".t" -type "double3" 2.0547211170196533 -34.92498882114887 1.5823587775230408 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "fbcfbd42-9065-4ee5-b778-71c97b7f098f";
	setAttr ".t" -type "double3" 0.8936494588851929 -29.1180819272995 6.159190833568573 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "33165113-0d4d-4a3d-82ad-062a6ae12694";
	setAttr ".t" -type "double3" -2.6430130004882812 -0.060561299324035645 -8.858744986355305 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "6ffc5661-30c2-4d99-a9df-0c285b164899";
	setAttr ".t" -type "double3" 4.6923041343688965 -35.531966388225555 -0.19529759883880615 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "a8ad5032-ed28-42fd-b18e-2096d0ab2f57";
	setAttr ".t" -type "double3" -1.6310006380081177 -36.751341819763184 -1.515912264585495 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "cfcd3f33-64d4-4755-b48c-52c883cd56bd";
	setAttr ".t" -type "double3" 0.09597837924957275 23.003731667995453 1.6774069517850876 ;
createNode joint -n "neck" -p "spine";
	rename -uid "c4072933-1993-4799-94e2-798e1288e4ad";
	setAttr ".t" -type "double3" -4.6182781457901 22.48789668083191 12.033951468765736 ;
createNode joint -n "head" -p "neck";
	rename -uid "e8e121c1-24dc-465a-a673-e550fe0fdb43";
	setAttr ".t" -type "double3" -2.1364718675613403 10.469883680343628 4.3089717626571655 ;
createNode joint -n "head_top" -p "head";
	rename -uid "f7b7f957-1597-44de-8ba1-efc40c009e05";
	setAttr ".t" -type "double3" -1.4549553394317627 10.48811674118042 6.299580633640289 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "91119f99-0699-45cd-bddc-c433828da79c";
	setAttr ".t" -type "double3" -0.6275147199630737 -1.7318278551101685 4.543562233448029 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "49e99a92-7593-4270-b2a3-5c2266e9069f";
	setAttr ".t" -type "double3" 23.649099469184875 10.219338536262512 8.266019821166992 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "3a97d9b0-7ca9-415b-9b85-ed16aeabc72d";
	setAttr ".t" -type "double3" 16.319911926984787 10.026299953460693 11.596238613128662 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "45c16d4f-7beb-4db0-a924-4b905e1f25bb";
	setAttr ".t" -type "double3" 0.456315279006958 1.6094386577606201 -11.069514602422714 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "8650ce10-0b2d-4961-9b58-463ccba94349";
	setAttr ".t" -type "double3" 21.590426564216614 10.818445682525635 -8.905929327011108 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "5aa84dee-900f-471d-9e1c-cee1ec8ddeb1";
	setAttr ".t" -type "double3" 18.37640330195427 9.48779582977295 -19.925298541784286 ;
