createNode transform -s -n "root_000002";
	rename -uid "ca43d65f-ffef-4c08-abbf-479aaf060cd7";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_000002";
	rename -uid "660c8a1a-d428-4d59-a5e5-d1e5212a18bf";
	setAttr ".t" -type "double3" -5.6655291467905045 -3.482436388731003 -0.20223306491971016 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "8aa90403-dd2f-408c-af9f-424ea3ba5dde";
	setAttr ".t" -type "double3" -9.694964811205864 -1.20917409658432 -0.4710858687758446 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "79e93a74-25a9-459b-9d1d-b771c268f32a";
	setAttr ".t" -type "double3" -0.4929080605506897 38.33077922463417 -23.65775750949979 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "bf043c85-0fef-463b-afcb-549802620413";
	setAttr ".t" -type "double3" 0.7379233837127686 39.435261487960815 33.86736437678337 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "e83abe67-50aa-4e36-896f-c08824b13432";
	setAttr ".t" -type "double3" 9.458546340465546 0.8073009550571442 1.7764021642506123 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "2582233c-a2d1-48d3-b811-5f3adf34350a";
	setAttr ".t" -type "double3" -1.172509789466858 40.868477523326874 27.981047332286835 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "6cfd1275-62a5-42f6-a633-e2d490946607";
	setAttr ".t" -type "double3" -9.067000076174736 34.65130031108856 17.289313673973083 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "16d56251-2587-400f-99a1-0de2e0ba4e71";
	setAttr ".t" -type "double3" 0.9232092648744583 -29.648400098085403 -3.104960825294256 ;
createNode joint -n "neck" -p "spine";
	rename -uid "c7baf9d9-40eb-4a60-ad18-b2a19f0c189c";
	setAttr ".t" -type "double3" 0.9791065007448196 -32.57972598075867 -5.854416266083717 ;
createNode joint -n "head" -p "neck";
	rename -uid "8da4fce4-17c5-4068-9ca5-251ecd48ad7e";
	setAttr ".t" -type "double3" -0.20746327936649323 -12.95960545539856 -2.9092609882354736 ;
createNode joint -n "head_top" -p "head";
	rename -uid "5c9da8f2-d5ea-47e7-9b6b-450b102e63d1";
	setAttr ".t" -type "double3" -0.25285184383392334 -12.604933977127075 -3.6349795758724213 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "414ec95c-f14b-4ee1-9496-c0538c91de70";
	setAttr ".t" -type "double3" -18.52620579302311 5.984383821487427 3.5336367785930634 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "e2028e7d-208d-48a1-8141-5f571909bbe1";
	setAttr ".t" -type "double3" -7.120969891548157 17.41141378879547 14.343413710594177 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "20533fcc-b23e-45f0-b068-78acbd9a44f4";
	setAttr ".t" -type "double3" -0.7910937070846558 18.484844267368317 -15.15171155333519 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "fafd4664-8fd3-48bd-93cb-4dac38728cdf";
	setAttr ".t" -type "double3" 18.361110612750053 6.2246620655059814 -2.256195992231369 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "d1519bab-834c-48e8-913f-40af424bb9f1";
	setAttr ".t" -type "double3" 15.124274790287018 22.81876802444458 -8.072561025619507 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "db05500f-1dfb-4be4-a61f-ba4e582c7f2b";
	setAttr ".t" -type "double3" -10.435165464878082 2.393236756324768 -19.896554946899414 ;
