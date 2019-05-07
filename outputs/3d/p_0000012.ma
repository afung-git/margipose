createNode transform -s -n "root_p_0000012";
	rename -uid "085b44bc-26ef-4516-9cfc-3aa002eeed22";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_p_0000012";
	rename -uid "2596bc6d-282e-4105-9397-95c32f69b47f";
	setAttr ".t" -type "double3" -4.095897823572159 3.449442610144615 0.6069615483283997 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "669e5823-aa18-4c09-8de6-64ab77f4a250";
	setAttr ".t" -type "double3" -4.050465673208237 -4.324155859649181 -3.1465906649827957 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "2e1b4af5-3b22-49b9-8edd-1b19aceae2af";
	setAttr ".t" -type "double3" 8.835676312446594 32.11048562079668 -42.47443936765194 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "26fe88ff-d832-4a30-8077-23fa85a2852d";
	setAttr ".t" -type "double3" -32.37789869308472 32.39928483963013 -2.9843389987945557 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "433dddfb-7d17-4af3-9281-8589702a6d60";
	setAttr ".t" -type "double3" 5.415063165128231 4.152962937951088 2.453199587762356 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "347efba4-1d97-423c-87e9-4ee02b4737af";
	setAttr ".t" -type "double3" -41.33316408842802 7.368074357509613 12.712414003908634 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "f2812d7e-6670-4821-8e59-252c6b30f514";
	setAttr ".t" -type "double3" -31.70763850212097 -15.048052929341793 2.7864694595336914 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "3082f599-0738-4911-abc3-20fda56d5968";
	setAttr ".t" -type "double3" 21.061985939741135 -14.369166269898415 -4.48032021522522 ;
createNode joint -n "neck" -p "spine";
	rename -uid "d96a1a3a-e3b1-4afe-8b67-45ca17f3dbc6";
	setAttr ".t" -type "double3" 22.55104035139084 -16.090553253889084 -9.580887854099274 ;
createNode joint -n "head" -p "neck";
	rename -uid "e9cb13f7-a6c5-435f-b2ad-d9fb498f6191";
	setAttr ".t" -type "double3" 9.893414378166199 -6.349962949752808 -3.3223122358322144 ;
createNode joint -n "head_top" -p "head";
	rename -uid "29eeeb58-babf-465d-8dab-9e772b4bfdb3";
	setAttr ".t" -type "double3" 10.026213526725769 -6.322896480560303 0.785374641418457 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "6c910b3b-62a5-4d15-b870-a7a587d07e8d";
	setAttr ".t" -type "double3" -13.27788233757019 -9.372314810752869 -2.34835147857666 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "34eaa9ef-640a-4a0f-ac8e-1bc5884aa737";
	setAttr ".t" -type "double3" -20.37058249115944 -11.077737808227539 -5.57650625705719 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "34566f98-b910-4f09-b7de-ce88d66c21cd";
	setAttr ".t" -type "double3" -20.47738805413246 8.21278989315033 1.5819981694221497 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "1f5cc867-6ef1-41f1-9b96-65c04db8d4eb";
	setAttr ".t" -type "double3" 5.151090025901794 14.544204622507095 3.230910748243332 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "1199ecce-a88f-4aa4-8e66-6d73b9aacb1a";
	setAttr ".t" -type "double3" -4.827800393104553 22.965456545352936 -12.54802718758583 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "f068eb4d-5463-40cf-a36c-ca51f978add3";
	setAttr ".t" -type "double3" 17.00422167778015 -13.983642682433128 -13.565589487552643 ;
