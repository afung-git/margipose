createNode transform -s -n "root_p_000005";
	rename -uid "c4b9465c-8bee-4ddb-acc8-4db84c9f1801";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_p_000005";
	rename -uid "b64844ec-ede6-49de-b113-3d536a8380d0";
	setAttr ".t" -type "double3" -1.1350156739354134 -1.8698036670684814 0.009113922715187073 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "120b55f9-a515-449b-b70b-cefcf0addcc7";
	setAttr ".t" -type "double3" 0.7201345637440681 0.09219758212566376 1.6324164345860481 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "25b60109-c0e3-4982-803e-d6efaa1a187e";
	setAttr ".t" -type "double3" -5.413069576025009 38.952310755848885 12.344133295118809 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "a1cf66b0-2573-4ff4-bfa5-2b2ba8d7d274";
	setAttr ".t" -type "double3" 2.5343388319015503 35.81884205341339 -1.2419909238815308 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "f8188c46-f7ec-427f-96dd-4ff123827f8c";
	setAttr ".t" -type "double3" -0.41309166699647903 -0.24354364722967148 -1.4538949355483055 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "8aab05ce-d0d8-4137-bc78-197e7d9cca1c";
	setAttr ".t" -type "double3" 0.1306973397731781 37.20352854579687 8.709479309618473 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "c47a35a2-30e3-4748-bfe2-50a4fcb5fbdb";
	setAttr ".t" -type "double3" -0.24531520903110504 32.51611888408661 3.819664567708969 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "cadccdcc-f409-410c-bbd4-750ed829b6eb";
	setAttr ".t" -type "double3" 2.318042330443859 -19.527991116046906 -4.952719807624817 ;
createNode joint -n "neck" -p "spine";
	rename -uid "aef74831-0172-4f88-9ed1-a09ef27e187b";
	setAttr ".t" -type "double3" 2.4363819509744644 -21.541278064250946 -6.30468912422657 ;
createNode joint -n "head" -p "neck";
	rename -uid "2fb32950-fe5a-4890-866b-e0925c21c9ad";
	setAttr ".t" -type "double3" 0.10984651744365692 -8.961033821105957 -3.035743534564972 ;
createNode joint -n "head_top" -p "head";
	rename -uid "bfd7143b-f185-470c-864e-6cd0236ed5fc";
	setAttr ".t" -type "double3" 5.897212401032448 -9.869158267974854 -6.199371814727783 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "12675da8-e767-48bd-8b9b-c165dbbae637";
	setAttr ".t" -type "double3" 0.9348772466182709 2.7658581733703613 2.7783624827861786 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "95bafa8b-fa67-48f4-ad3f-05dc3e3e618b";
	setAttr ".t" -type "double3" -2.9470693320035934 18.157270550727844 11.989157646894455 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "f73e1b64-f65d-403c-a536-64c35d4a1aa8";
	setAttr ".t" -type "double3" 1.4559347182512283 1.0791599750518799 -7.527321949601173 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "b4983d68-c364-4409-b5fa-277332e6cece";
	setAttr ".t" -type "double3" -0.48280321061611176 1.6683876514434814 -1.0804757475852966 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "23e54b08-6135-4c75-9fe0-25c897d6d31c";
	setAttr ".t" -type "double3" -3.9914023131132126 17.34199821949005 7.280242443084717 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "0c36100e-7591-4ba4-9b16-aa8a5ee48238";
	setAttr ".t" -type "double3" 0.06826333701610565 6.029972434043884 -0.2950366586446762 ;
