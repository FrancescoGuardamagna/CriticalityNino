
class wrapper_best_parameters:

    def __init__(self):

        self.bestParmasDictionaryZC60={0.5:{'Nx': 949, 'alpha': 0.6565476024987439, 
        'density_W': 0.8192310561891545, 'sp': 0.13029733288511122, 'input_scaling': 1.9326402017980753, 
        'bias_scaling': 0.23026138970542515, 'reservoir_scaling': 1.8203370643757164, 
        'regularization': 0.09663665204810813},0.7:{'Nx': 857, 'alpha': 0.2153849592084341, 
        'density_W': 0.18516948302267439, 'sp': 0.2197965266811174, 'input_scaling': 1.144639656745684, 
        'bias_scaling': -1.7021967147224735, 'reservoir_scaling': 1.3901742101263628, 
        'regularization': 0.09303398185940587},1:{'Nx': 984, 'alpha': 0.2441049413139878, 'density_W': 0.7840504473350758, 
        'sp': 0.7844360441822062, 'input_scaling': 1.834235687668562, 'bias_scaling': -0.0016252006541570127, 
        'reservoir_scaling': 0.3511922380515522, 'regularization': 0.09491336817191884}}

        self.bestParmasDictionaryZC40={0.5:{'Nx': 962, 'alpha': 0.5263707351326109, 
        'density_W': 0.6639490819598365, 'sp': 0.6110232061753674, 'input_scaling': 1.9435768747451367, 
        'bias_scaling': 0.7280031026973903, 'reservoir_scaling': 0.24861648097355327, 
        'regularization': 0.09647732856099168},0.7:{'Nx': 952, 'alpha': 0.399626393802332, 
        'density_W': 0.35943815097229326, 'sp': 0.8556668421192987, 'input_scaling': 0.7398462323527635, 
        'bias_scaling': 1.9994533432189874, 'reservoir_scaling': 0.19959970613946848, 
        'regularization': 0.09567956785059865},1:{'Nx': 976, 'alpha': 0.28418309412231074, 
        'density_W': 0.19009433237617365, 'sp': 0.8085458690817477, 'input_scaling': 1.814909932062059, 
        'bias_scaling': -1.0904891505734646, 'reservoir_scaling': 1.3640682551803185, 
        'regularization': 0.09815025529712594}}

        self.bestParamsDictionaryZC20={0.5:{'Nx': 550, 'alpha': 0.42679425837196683, 
        'density_W': 0.10108854249177723, 'sp': 0.718814179656705, 'input_scaling': 0.6251047470277503, 
        'bias_scaling': 0.11710066494880592, 'reservoir_scaling': 0.8308679932865127, 
        'regularization': 0.09459643945201598},0.7:{'Nx': 970, 'alpha': 0.5062500832382636, 
        'density_W': 0.7877084433126714, 'sp': 0.9217486844195029, 'input_scaling': 0.8408128871174596, 
        'bias_scaling': 1.1101179444792226, 'reservoir_scaling': 0.2911635665034175, 
        'regularization': 0.0970363390927365},1:{'Nx': 966, 'alpha': 0.5193853232405021, 
        'density_W': 0.9202530239499744, 'sp': 0.2043187327783506, 'input_scaling': 1.7830429996206114, 
        'bias_scaling': -1.848102161723374, 'reservoir_scaling': 0.34929586461784345, 
        'regularization': 0.09483350135802661}}

        self.best_parameters_low_Nino_34={'Nx': 205, 'alpha': 0.8659545131784361, 
        'density_W': 0.23187576668735968, 'sp': 0.7292985631213502, 
        'input_scaling': 0.10469258983273283, 'bias_scaling': 0.4030388483464956, 
        'reservoir_scaling': 0.361747364504155, 'regularization': 0.008175605454283402}

        self.best_parameters_CESM_low_Nino_3={'Nx': 286, 'alpha': 0.8994422717761436, 
        'density_W': 0.17186595774924868, 'sp': 0.5671706512588007, 
        'input_scaling': 0.10148419753179619, 'bias_scaling': -0.517313935588865, 
        'reservoir_scaling': 0.8309924921249283, 'regularization': 0.0045701278908332355}

        self.best_parameters_high_Nino_34={'Nx': 717, 'alpha': 0.9729955319960034, 
        'density_W': 0.8071276173109384, 'sp': 0.7095153084559543, 
        'input_scaling': 0.10618905675831158, 'bias_scaling': -0.6137977486968632, 
        'reservoir_scaling': 1.5895923805670842, 'regularization': 0.009796744807800024}

        self.best_parameters_real={'Nx': 109, 'alpha': 0.8770492722520004, 'density_W': 0.8829301919846817, 
        'sp': 0.3551009920598088, 'input_scaling': 0.10020807076687648, 'bias_scaling': -0.2914598835399332, 
        'reservoir_scaling': 1.7103727059794105, 'regularization': 0.002976367105475108}

        self.best_parameters_Jin_100={'Nx': 628, 'alpha': 0.8702621014094571, 
        'density_W': 0.7614561445000985, 'sp': 0.7705940612804938, 'input_scaling': 1.9117732577851665, 
        'bias_scaling': -0.6941816886181638, 'reservoir_scaling': 0.4691052817428416, 
        'regularization': 0.0009563165438334443,'training_weights': 1}

        self.best_parameters_Jin_50={'Nx': 424, 'alpha': 0.8714250901400902, 'density_W': 0.9294312206310186, 
        'sp': 0.2910758290742638, 'input_scaling': 1.6812253644961817, 'bias_scaling': 1.0521707285687303, 
        'reservoir_scaling': 1.4832596329210348, 'regularization': 0.004053683779393836}

        self.best_parameters_Jin_30={'Nx': 337, 'alpha': 0.3474555696995316, 'density_W': 0.5661640023224153, 
        'sp': 0.9484544675129903, 'input_scaling': 1.7730253491262682, 'bias_scaling': 0.7614638324126203, 
        'reservoir_scaling': 1.350919588101399, 'regularization': 0.006462238728029472}

        self.best_parameters_Jin_40={'Nx': 460, 'alpha': 0.24575028880035577, 'density_W': 0.35878697456211717, 'sp': 0.5275989486236409, 
        'input_scaling': 1.881011591166569, 'bias_scaling': 1.7722324524333253, 'reservoir_scaling': 1.436003287017181, 
        'regularization': 0.006458879956822417}

        self.best_parameters_Jin_1_variable={'Nx': 457, 'alpha': 0.560321341338875, 'density_W': 0.42666214685509607, 
        'sp': 0.9238767412355287, 'input_scaling': 1.57522151125071, 'bias_scaling': 1.7125556122967396, 
        'reservoir_scaling': 1.277779851459917, 'regularization': 0.00015765761926127964}

        self.best_parameters_Jin_xy={'Nx': 537, 'alpha': 0.999768102725525, 'density_W': 0.6967692160105639, 
        'sp': 0.7534269574793903, 'input_scaling': 1.8816700198573542, 'bias_scaling': 1.9995997267935672, 
        'reservoir_scaling': 1.3185439102478718, 'regularization': 0.0004899328272211062}

        self.best_parameters_Jin_xz={'Nx': 932, 'alpha': 0.6276248686412241, 'density_W': 0.5741832744556432, 
        'sp': 0.979444097169754, 'input_scaling': 1.9630789152263246, 'bias_scaling': -1.2175609835967733, 
        'reservoir_scaling': 1.9998890875082793, 'regularization': 0.003269650239628797}

        self.best_parameters_CESM_Low_1200_1220={'Nx': 126, 'alpha': 0.5073337139113102, 
        'density_W': 0.13849683970668517, 'sp': 0.21454891286427194, 
        'input_scaling': 0.10113255883088088, 'bias_scaling': -0.2197602876626853, 
        'reservoir_scaling': 0.10018236080857579, 'regularization': 0.008190071377254741}

        self.best_parameters_CESM_Low_1210_1230={'Nx': 124, 'alpha': 0.817352835911591, 
        'density_W': 0.834992049359146, 'sp': 0.42565003115132877, 
        'input_scaling': 0.14506541256661662, 'bias_scaling': 0.020316548917333005, 
        'reservoir_scaling': 1.2458592316396326, 'regularization': 0.00539034923504601}

        self.best_parameters_CESM_Low_1220_1240={'Nx': 60, 'alpha': 0.6722018064633504, 
        'density_W': 0.34819343554958615, 'sp': 0.33890264462009667, 
        'input_scaling': 0.10071940582487063, 'bias_scaling': -0.0930275891159094, 
        'reservoir_scaling': 1.8707767950905798, 'regularization': 0.005065130199223591}

        self.best_parameters_CESM_Low_1230_1250={'Nx': 224, 'alpha': 0.996068154449349, 
        'density_W': 0.5012473856303069, 'sp': 0.2710584628781344, 'input_scaling': 0.17342275595981846, 
        'bias_scaling': 1.650766906500155, 'reservoir_scaling': 1.855794560068629, 
        'regularization': 0.004778582780305963}

        self.best_parameters_CESM_Low_1240_1260={'Nx': 247, 'alpha': 0.9171397430120328, 
        'density_W': 0.3792412404489832, 
        'sp': 0.22942427716168654, 'input_scaling': 0.1610341393230262, 
        'bias_scaling': -0.0338575644907681, 'reservoir_scaling': 0.7007066144522408, 
        'regularization': 0.00739001621935}

        self.best_parameters_CESM_Low_1250_1270={'Nx': 24, 'alpha': 0.9847198011205962, 
        'density_W': 0.6201798964953541, 'sp': 0.591894513192072, 
        'input_scaling': 0.22894597974618044, 'bias_scaling': -0.058015051743895374, 
        'reservoir_scaling': 0.10112009494811369, 'regularization': 0.00960488488707627}

        self.best_parameters_CESM_Low_1260_1280={'Nx': 275, 'alpha': 0.9991683219814467, 
        'density_W': 0.8010892238972653, 'sp': 0.4063486934167766, 
        'input_scaling': 0.10307825816235458, 'bias_scaling': -1.71559550862255, 
        'reservoir_scaling': 0.18103167939667758, 'regularization': 0.0028057194870476313}

        self.best_parameters_CESM_Low_1270_1290={'Nx': 574, 'alpha': 0.9834738030808302, 
        'density_W': 0.6304124239778608, 'sp': 0.21158013777149554, 
        'input_scaling': 0.14467721103460351, 'bias_scaling': -0.01201399919785641, 
        'reservoir_scaling': 1.2850088950888958, 'regularization': 0.009562128366539127}

        self.best_parameters_CESM_Low_1280_1300={'Nx': 133, 'alpha': 0.9981578556547643, 
        'density_W': 0.49727048117972383, 'sp': 0.24664853276913, 
        'input_scaling': 0.13897876827177755, 'bias_scaling': 0.004618418719796458, 
        'reservoir_scaling': 0.539413945764406, 'regularization': 0.006286223402610647}

        self.best_parameters_CESM_High_200_220={'Nx': 101, 'alpha': 0.6487268203388293, 
        'density_W': 0.2708544464674532, 'sp': 0.1492772200442672, 
        'input_scaling': 0.10174007644190143, 'bias_scaling': -0.035326844050010006, 
        'reservoir_scaling': 0.24811314688815067, 'regularization': 0.0016696589287478994}

        self.best_parameters_CESM_High_210_230={'Nx': 78, 'alpha': 0.9629171659854879, 
        'density_W': 0.5057397180579637, 'sp': 0.31044918696843626, 
        'input_scaling': 0.10106999725064891, 'bias_scaling': -0.0739615238023289, 
        'reservoir_scaling': 1.199127772562341, 'regularization': 0.003003208746150312}

        self.best_parameters_CESM_High_220_240={'Nx': 21, 'alpha': 0.9878812145906838, 
        'density_W': 0.8078039545507321, 'sp': 0.3811035422580579, 
        'input_scaling': 0.10265858851117263, 'bias_scaling': 0.016209610353395687, 
        'reservoir_scaling': 0.3996664033132862, 'regularization': 0.009739620117636828}

        self.best_parameters_CESM_High_230_250={'Nx': 83, 'alpha': 0.9177254865386266, 
        'density_W': 0.5962912945291686, 'sp': 0.709782184961759, 
        'input_scaling': 0.10603327638562766, 'bias_scaling': 1.3881248853100259, 
        'reservoir_scaling': 1.3206180075915928, 'regularization': 0.008950786367828558}

        self.best_parameters_CESM_High_240_260={'Nx': 22, 'alpha': 0.8257803335350848, 
        'density_W': 0.45456146294467337, 'sp': 0.563657594153085, 
        'input_scaling': 0.1009208383534489, 'bias_scaling': 0.16570941344257634, 
        'reservoir_scaling': 1.5490403975631786, 'regularization': 0.008696278463596614}

        self.best_parameters_CESM_High_250_270={'Nx': 45, 'alpha': 0.7129736434115188, 
        'density_W': 0.5425572639370192, 'sp': 0.3793623401514651, 
        'input_scaling': 0.1000666400443272, 'bias_scaling': 1.8724428406468454, 
        'reservoir_scaling': 0.3318495803310887, 'regularization': 0.008586906930458913}

        self.best_parameters_CESM_High_260_280={'Nx': 86, 'alpha': 0.9549260511347447, 
        'density_W': 0.6848152057999279, 'sp': 0.25416826272852533, 
        'input_scaling': 0.10048373467189932, 'bias_scaling': 0.2788584009324868, 
        'reservoir_scaling': 0.8845144073618673, 'regularization': 0.009349219273173495}

        self.best_parameters_CESM_High_270_290={'Nx': 39, 'alpha': 0.913534765375918, 
        'density_W': 0.42022315506986396, 'sp': 0.36715721545188484, 
        'input_scaling': 0.10120796242980443, 'bias_scaling': 0.2873912582105884, 
        'reservoir_scaling': 0.4026370685720706, 'regularization': 0.008118833601173474}

        self.best_parameters_CESM_High_280_300={'Nx': 68, 'alpha': 0.6795113182603734, 
        'density_W': 0.9247933709678261, 'sp': 0.17829549225904803, 
        'input_scaling': 0.10048241677432218, 'bias_scaling': 0.04822072651950398, 
        'reservoir_scaling': 1.3525246599256737, 'regularization': 0.0070253208550816616}

        self.best_parameters_Real_1960_1980={'Nx': 111, 'alpha': 0.9855006239505167, 
        'density_W': 0.4316768555025382, 'sp': 0.23692164041219094, 
        'input_scaling': 0.10120199351889195, 'bias_scaling': 0.1280419218440886, 
        'reservoir_scaling': 0.8923133466045878, 'regularization': 0.007789728415967539}

        self.best_parameters_Real_1970_1990={'Nx': 997, 'alpha': 0.9482889825269339, 
        'density_W': 0.8648740674576165, 'sp': 0.9994545072275844, 
        'input_scaling': 1.9598707259951533, 'bias_scaling': 1.2953049578997122, 
        'reservoir_scaling': 1.2167413322208347, 'regularization': 0.0038253232773043035}

        self.best_parameters_Real_1980_2000={'Nx': 5, 'alpha': 0.7916659576550927, 
        'density_W': 0.28590880834645765, 'sp': 0.13698379004475492, 
        'input_scaling': 0.10131997848974929, 'bias_scaling': 0.17345869578163817, 
        'reservoir_scaling': 0.7141171762370535, 'regularization': 0.006852216692229055}

        self.best_parameters_Real_1990_2010={'Nx': 25, 'alpha': 0.9905230414280027, 
        'density_W': 0.8843493754853018, 'sp': 0.38892603161025047, 
        'input_scaling': 0.10873831008522135, 'bias_scaling': -0.416056303558297, 
        'reservoir_scaling': 0.927656607203041, 'regularization': 0.00987476541916535}

        self.best_parameters_Real_2000_2020={'Nx': 21, 'alpha': 0.9455889081319292, 
        'density_W': 0.9370017816029508, 'sp': 0.5835275258191095, 
        'input_scaling': 0.10124467509095803, 'bias_scaling': -0.8771689192752262, 
        'reservoir_scaling': 1.920673784177829, 'regularization': 0.005473315181579288}

        self.best_parameters_CESM_Low_1200_1240_Test_1240_1260={'Nx': 150, 'alpha': 0.5599564349858908, 
        'density_W': 0.9211277120079905, 'sp': 0.13616803083948628, 'input_scaling': 0.10234989614200399, 
        'bias_scaling': -0.4503683552466704, 'reservoir_scaling': 1.5994556017789798, 'regularization': 0.005619188619383252}

        self.best_parameters_CESM_Low_1210_1250_Test_1250_1270={'Nx': 120, 'alpha': 0.6043201641724775, 
        'density_W': 0.10191167761752701, 'sp': 0.18222839657261958, 'input_scaling': 0.1038093123532324, 
        'bias_scaling': -0.2978327724499683, 'reservoir_scaling': 0.6952693834389728, 
        'regularization': 0.0040944007660195505}

        self.best_parameters_CESM_Low_1200_1240_Test_1280_1300={'Nx': 275, 'alpha': 0.7845329831093408, 
        'density_W': 0.4585423636449583, 'sp': 0.714700378275293, 'input_scaling': 0.10040606324694465, 
        'bias_scaling': 1.8687775663791506, 'reservoir_scaling': 0.19435479007988088, 
        'regularization': 0.009258007009825842}

        self.best_parameters_CESM_Low_1210_1250_Test_1280_1300={'Nx': 147, 'alpha': 0.9196432777748416, 
        'density_W': 0.2263433057854621, 'sp': 0.614443326043111, 'input_scaling': 0.10394733725156324, 
        'bias_scaling': -0.8527416438163862, 'reservoir_scaling': 0.5871449486247081, 'regularization': 0.00944613246072714}

        self.best_parameters_CESM_Low_1220_1260_Test_1280_1300={'Nx': 155, 'alpha': 0.9508491367690869, 
        'density_W': 0.8915411045577142, 'sp': 0.38774805087020314, 'input_scaling': 0.10033501635500974, 
        'bias_scaling': 0.390235522537585, 'reservoir_scaling': 1.9943038805684516, 'regularization': 0.00401960593197699}

        self.best_parameters_CESM_Low_1230_1270_Test_1280_1300={'Nx': 891, 'alpha': 0.6589415121097127, 
        'density_W': 0.955019255660678, 'sp': 0.13826639989513365, 'input_scaling': 0.10080033356923479, 
        'bias_scaling': 0.019847614407693254, 'reservoir_scaling': 0.8115086863026381, 'regularization': 0.00931143002797632}

        self.best_parameters_CESM_Low_1240_1280_Test_1280_1300={'Nx': 245, 'alpha': 0.8067782324527368, 'density_W': 0.7193203103205295, 
        'sp': 0.7160297828610069, 'input_scaling': 0.10449657558702705, 'bias_scaling': 1.0301082609334893, 
        'reservoir_scaling': 0.4773586770336411, 'regularization': 0.009698362393376226}


        self.best_parameters_CESM_Low_1200_1240_Test_1280_1300_Full={'Nx': 59, 'alpha': 0.8112948997948255, 
        'density_W': 0.4544451223164093, 'sp': 0.41151320407444786, 'input_scaling': 0.10334535715879871, 
        'bias_scaling': -0.5693638468097106, 'reservoir_scaling': 1.5900252513315039, 'regularization': 0.005032626076124455}

        self.best_parameters_CESM_Low_1210_1250_Test_1280_1300_Full={'Nx': 307, 'alpha': 0.8430862979168325, 
        'density_W': 0.35042682773174494, 'sp': 0.4311461560310617, 'input_scaling': 0.1005694191245999, 
        'bias_scaling': -0.3897737605983941, 'reservoir_scaling': 1.0830909375981805, 'regularization': 0.008946233864631819}

        self.best_parameters_CESM_Low_1220_1260_Test_1280_1300_Full={'Nx': 358, 'alpha': 0.9805334926213428, 
        'density_W': 0.49721180515785607, 'sp': 0.49153107295305215, 'input_scaling': 0.10128011232083048,
        'bias_scaling': 0.023726788089508198, 'reservoir_scaling': 1.6963686392108246, 'regularization': 0.004185290416285946}

        self.best_parameters_CESM_Low_1230_1270_Test_1280_1300_Full={'Nx': 113, 'alpha': 0.8334137985249683, 
        'density_W': 0.47367710109735234, 'sp': 0.814236852950843, 'input_scaling': 0.10792824575612511, 
        'bias_scaling': 1.809600092352434, 'reservoir_scaling': 1.3266149894525203, 'regularization': 0.0022407110403983546}

        self.best_parameters_CESM_Low_1240_1280_Test_1280_1300_Full={'Nx': 384, 'alpha': 0.997650701904844, 
        'density_W': 0.8945213101609576, 'sp': 0.8988600937093097, 'input_scaling': 0.10044180494624466, 
        'bias_scaling': 0.8629526951497157, 'reservoir_scaling': 0.9005965955118858, 'regularization': 0.008056016991474502}

        self.best_parameters_CESM_Low_1200_1240_Test_1280_1300_Lead_3={'Nx': 139, 'alpha': 0.9630035826871159, 
        'density_W': 0.6239544918454799, 'sp': 0.746240043712693, 'input_scaling': 0.10106793319541295, 
        'bias_scaling': -0.3137071824405452, 'reservoir_scaling': 1.6968800211648711, 'regularization': 0.007873417639447938}

        self.best_parameters_CESM_Low_1200_1240_Test_1280_1300_Lead_6={'Nx': 72, 'alpha': 0.9462355828046751, 
        'density_W': 0.39524720243836464, 'sp': 0.9761849279415739, 'input_scaling': 0.10119270139866478, 
        'bias_scaling': -0.41964980743985236, 'reservoir_scaling': 1.56933870994606, 'regularization': 0.0093975292521568}

        self.best_parameters_CESM_Low_1200_1240_Test_1280_1300_Lead_9={'Nx': 34, 'alpha': 0.5837870025952672, 
        'density_W': 0.32291444787719115, 'sp': 0.4192965016168418, 'input_scaling': 0.10030567261783778, 
        'bias_scaling': -1.8590534269648578, 'reservoir_scaling': 0.16695098333143338, 'regularization': 0.009727845448798041}

        self.best_parameters_CESM_Low_1210_1250_Test_1280_1300_Lead_3={'Nx': 47, 'alpha': 0.9276044507973277, 
        'density_W': 0.503235851947348, 'sp': 0.5027235990490173, 'input_scaling': 0.14784566213982672, 
        'bias_scaling': -0.19357853074267534, 'reservoir_scaling': 1.6537909922631315, 'regularization': 0.008458060235731068}

        self.best_parameters_CESM_Low_1210_1250_Test_1280_1300_Lead_6={'Nx': 53, 'alpha': 0.668536747406767, 
        'density_W': 0.8559116949194329, 'sp': 0.41875019341043584, 'input_scaling': 0.10169193848539863, 
        'bias_scaling': 1.0833242260117564, 'reservoir_scaling': 1.9938100316097318, 'regularization': 0.006343601379169484}

        self.best_parameters_CESM_Low_1210_1250_Test_1280_1300_Lead_9={'Nx': 76, 'alpha': 0.8721712886375416, 
        'density_W': 0.5377420458515589, 'sp': 0.4819516756755673, 'input_scaling': 0.10009471674278446, 
        'bias_scaling': -0.49348996884896323, 'reservoir_scaling': 1.4039443088720107, 'regularization': 0.0033159136160944526}

        self.best_parameters_CESM_Low_1220_1260_Test_1280_1300_Lead_3={'Nx': 86, 'alpha': 0.8917829515477885, 
        'density_W': 0.38177947045174343, 'sp': 0.7216490554785773, 'input_scaling': 0.10193744696130612, 
        'bias_scaling': -1.4076413240259806, 'reservoir_scaling': 1.342706481226022, 'regularization': 0.008634191411860285}

        self.best_parameters_CESM_Low_1220_1260_Test_1280_1300_Lead_6={'Nx': 114, 'alpha': 0.4762922377109316,
        'density_W': 0.13642233812562138, 'sp': 0.28417416824872144, 'input_scaling': 0.13658480773148407, 
        'bias_scaling': 0.21037754689739724, 'reservoir_scaling': 0.2999515847804529, 'regularization': 0.00901140604347098}

        self.best_parameters_CESM_Low_1220_1260_Test_1280_1300_Lead_9={'Nx': 241, 'alpha': 0.5921534326876394, 
        'density_W': 0.4398043722144315, 'sp': 0.10006724615686628, 'input_scaling': 0.10030291057019039, 
        'bias_scaling': -0.08722990426326162, 'reservoir_scaling': 1.626961107591027, 'regularization': 0.003911855187010595}

        self.best_parameters_CESM_Low_1230_1270_Test_1280_1300_Lead_3={'Nx': 524, 'alpha': 0.9764067073497368, 
        'density_W': 0.47478322064398426, 'sp': 0.5487845419524325, 'input_scaling': 0.10047399943711388, 
        'bias_scaling': 0.043765608868701865, 'reservoir_scaling': 0.9122138035153854, 'regularization': 0.0038655357539928584}

        self.best_parameters_CESM_Low_1230_1270_Test_1280_1300_Lead_6={'Nx': 25, 'alpha': 0.44376456240707474, 
        'density_W': 0.5006084869255569, 'sp': 0.2775812410462077, 'input_scaling': 0.25755475718046444, 
        'bias_scaling': -0.10734132140441614, 'reservoir_scaling': 0.7093006492170202, 'regularization': 0.007690337512674024}

        self.best_parameters_CESM_Low_1230_1270_Test_1280_1300_Lead_9={'Nx': 47, 'alpha': 0.633375647357786, 
        'density_W': 0.6997860865412012, 'sp': 0.31733166942476854, 'input_scaling': 0.10779479377534278, 
        'bias_scaling': 0.3872929817582528, 'reservoir_scaling': 1.002068435178059, 'regularization': 0.003524894985274004}

        self.best_parameters_CESM_Low_1240_1280_Test_1280_1300_Lead_3={'Nx': 622, 'alpha': 0.9124853842937469, 
        'density_W': 0.5530401137309733, 'sp': 0.5770429998317833, 'input_scaling': 0.10023307458975916, 
        'bias_scaling': 0.5883753696041403, 'reservoir_scaling': 1.1622552505052435, 'regularization': 0.007452504577295216}

        self.best_parameters_CESM_Low_1240_1280_Test_1280_1300_Lead_6={'Nx': 120, 'alpha': 0.9513003711141272, 
        'density_W': 0.5706807054427859, 'sp': 0.7627917540967223, 'input_scaling': 0.10289139306179146, 
        'bias_scaling': -0.12312049664164282, 'reservoir_scaling': 1.2776422486564172, 'regularization': 0.007393982290322575}

        self.best_parameters_CESM_Low_1240_1280_Test_1280_1300_Lead_9={'Nx': 314, 'alpha': 0.5429086984164798, 
        'density_W': 0.33296675131644526, 'sp': 0.14671197612771752, 'input_scaling': 0.10012504743169862, 
        'bias_scaling': -0.975440562020129, 'reservoir_scaling': 1.5965003745032504, 'regularization': 0.003014018248457183}

        self.best_parameters_CESM_Low_1200_1240_Test_1280_1300_Lead_3_full={'Nx': 82, 'alpha': 0.8062279799384893, 
        'density_W': 0.8507151925168132, 'sp': 0.5443498702050209, 'input_scaling': 0.10013831803703147, 
        'bias_scaling': -0.27832450336547826, 'reservoir_scaling': 1.049966789766604, 'regularization': 0.0028933281210933018}

        self.best_parameters_CESM_Low_1200_1240_Test_1280_1300_Lead_6_full={'Nx': 122, 'alpha': 0.9910502311713449, 
        'density_W': 0.3775901869674067, 'sp': 0.7303697731306426, 'input_scaling': 0.10064683206770826, 
        'bias_scaling': -1.7868576132770952, 'reservoir_scaling': 0.9251713125577463, 'regularization': 0.00692112819893088}

        self.best_parameters_CESM_Low_1200_1240_Test_1280_1300_Lead_9_full={'Nx': 135, 'alpha': 0.9608594899607712, 
        'density_W': 0.9329903032844837, 'sp': 0.9997448455546177, 'input_scaling': 0.10170523407104745, 
        'bias_scaling': -1.9168937904416734, 'reservoir_scaling': 0.2220933073867742, 'regularization': 0.009209178636327888}

        self.best_parameters_CESM_Low_1210_1250_Test_1280_1300_Lead_3_full={'Nx': 348, 'alpha': 0.9984378247838245, 
        'density_W': 0.21450594551140528, 'sp': 0.6145978849528184, 'input_scaling': 0.10069133189923905, 
        'bias_scaling': 1.3687187948484112, 'reservoir_scaling': 0.6264553242855224, 'regularization': 0.008930648432208973}

        self.best_parameters_CESM_Low_1210_1250_Test_1280_1300_Lead_6_full={'Nx': 197, 'alpha': 0.9976008051779179, 
        'density_W': 0.21993154893820488, 'sp': 0.28252123549791547, 'input_scaling': 0.10038912908979339, 
        'bias_scaling': -0.5976619225751352, 'reservoir_scaling': 0.33917639575923847, 'regularization': 0.008713660296820093}

        self.best_parameters_CESM_Low_1210_1250_Test_1280_1300_Lead_9_full={'Nx': 455, 'alpha': 0.9646295216160775, 
        'density_W': 0.2444808709134262, 'sp': 0.7905425259248033, 'input_scaling': 0.10475361355695341, 
        'bias_scaling': 1.6213857951155497, 'reservoir_scaling': 0.4107303132490268, 'regularization': 0.009813544107464711}

        self.best_parameters_CESM_Low_1220_1260_Test_1280_1300_Lead_3_full={'Nx': 319, 'alpha': 0.9763687600428557, 
        'density_W': 0.6372788823784092, 'sp': 0.6295041400184163, 'input_scaling': 0.10024665130755951, 
        'bias_scaling': -0.004805004314361677, 'reservoir_scaling': 0.5589531055051922, 'regularization': 0.007601406010151331}

        self.best_parameters_CESM_Low_1220_1260_Test_1280_1300_Lead_6_full={'Nx': 475, 'alpha': 0.6344044155242501, 
        'density_W': 0.3315899434642161, 'sp': 0.2648262837842043, 'input_scaling': 0.12492202705335008, 
        'bias_scaling': -0.03120555675273451, 'reservoir_scaling': 0.8752944198759791, 'regularization': 0.009720657341373135}

        self.best_parameters_CESM_Low_1220_1260_Test_1280_1300_Lead_9_full={'Nx': 59, 'alpha': 0.5607596402651761, 
        'density_W': 0.1796423606569249, 'sp': 0.25927488407926286, 'input_scaling': 0.17126455358188294, 
        'bias_scaling': 0.7008228315219585, 'reservoir_scaling': 0.34265807625668365, 'regularization': 0.008179953821461824}

        self.best_parameters_CESM_Low_1230_1270_Test_1280_1300_Lead_3_full={'Nx': 528, 'alpha': 0.9453768465069115, 
        'density_W': 0.8761818339758185, 'sp': 0.531277049156287, 'input_scaling': 0.13804168527070293, 
        'bias_scaling': -0.009915038586385993, 'reservoir_scaling': 1.9237171190011477, 'regularization': 0.0060963429797855845}

        self.best_parameters_CESM_Low_1230_1270_Test_1280_1300_Lead_6_full={'Nx': 63, 'alpha': 0.6137630044897799, 
        'density_W': 0.3225006668686193, 'sp': 0.44109111526114747, 'input_scaling': 0.10111635799268388, 
        'bias_scaling': 0.4188475956843525, 'reservoir_scaling': 0.7733982034448308, 'regularization': 0.0013840236408412762}

        self.best_parameters_CESM_Low_1230_1270_Test_1280_1300_Lead_9_full={'Nx': 241, 'alpha': 0.9689074779802986, 
        'density_W': 0.8094571238737454, 'sp': 0.8480770491240166, 'input_scaling': 0.10225356736443872,
         'bias_scaling': 1.2279900510301507, 'reservoir_scaling': 1.5868607908867998, 'regularization': 0.009033616846929778}

        self.best_parameters_CESM_Low_1240_1280_Test_1280_1300_Lead_3_full={'Nx': 778, 'alpha': 0.9613472069416462, 
        'density_W': 0.48277064251467794, 'sp': 0.9671469784073804, 'input_scaling': 0.1331670672422331, 
        'bias_scaling': 1.3724072022867413, 'reservoir_scaling': 0.7730173503317687, 'regularization': 0.009314889858642804}

        self.best_parameters_CESM_Low_1240_1280_Test_1280_1300_Lead_6_full={'Nx': 538, 'alpha': 0.9990132718372823, 
        'density_W': 0.36408736897184824, 'sp': 0.6969524769603699, 'input_scaling': 0.10117825505223611, 
        'bias_scaling': -0.5386647162136824, 'reservoir_scaling': 1.6073646415757443, 'regularization': 0.006943798754914692}

        self.best_parameters_CESM_Low_1240_1280_Test_1280_1300_Lead_9_full={'Nx': 254, 'alpha': 0.8171446041171812, 
        'density_W': 0.6306679337352954, 'sp': 0.7762485599402301, 'input_scaling': 0.10594479345412201, 
        'bias_scaling': 0.9689919641202962, 'reservoir_scaling': 0.5376161181833488, 'regularization': 0.005657083853894013}

        self.best_parameters_CESM_Low_1200_1240_Test_1280_1300_WP_full={'Nx': 954, 'alpha': 0.6994036294996804, 
        'density_W': 0.5127658556122748, 'sp': 0.1659241548920617, 'input_scaling': 0.10045445132508644, 
        'bias_scaling': 0.0658223655039914, 'reservoir_scaling': 1.0599759362214178, 'regularization': 0.00867526402084805}

        self.best_parameters_CESM_Low_1200_1240_Test_1280_1300_WP_Thermocline_splitted={'Nx': 249, 'alpha': 0.6425716239294408, 
        'density_W': 0.13404340981310972, 'sp': 0.1856240349998886, 'input_scaling': 0.1000240686559881, 
        'bias_scaling': -0.04912385843200916, 'reservoir_scaling': 1.7459618156549208, 'regularization': 0.004278935522612829}

        self.best_parameters_CESM_Low_1200_1240_Test_1280_1300_WP_All_data={'Nx': 324, 'alpha': 0.6809565463488232, 
        'density_W': 0.7062982352135396, 'sp': 0.188941800743926, 'input_scaling': 0.10128164789704297, 
        'bias_scaling': 0.01377607604743894, 'reservoir_scaling': 0.6548891944179224, 'regularization': 0.0077374711435175166}

        self.best_parameters_CESM_Low_1200_1240_Test_1280_1300_WP_full_All_data={'Nx': 273, 'alpha': 0.9991749457179155, 
        'density_W': 0.8353873076428247, 'sp': 0.4730283347777421, 'input_scaling': 0.10064395175027512, 
        'bias_scaling': -1.4166199348589232, 'reservoir_scaling': 1.1450416193083135, 'regularization': 0.0075103816445439035}

        self.best_parameters_CESM_Low_1200_1240_Test_1280_1300_WP_All_data_No_STW={'Nx': 88, 
        'alpha': 0.9997451930495383, 'density_W': 0.5685074659141357, 'sp': 0.5900758255792854, 
        'input_scaling': 0.10029283793238555, 'bias_scaling': -0.25232883026360636, 'reservoir_scaling': 1.037843764591142, 
        'regularization': 0.008488182644288073}

        self.best_parameters_CESM_Low_1200_1240_Test_1280_1300_WP_full_All_data_No_STW={'Nx': 408, 'alpha': 0.9947612567692443, 
        'density_W': 0.5405886886356909, 'sp': 0.4628255807585744, 'input_scaling': 0.10026641912638118, 
        'bias_scaling': 0.9724036603268039, 'reservoir_scaling': 1.4677298617069474, 'regularization': 0.006231085028401928}

        self.best_parameters_Low_Train_All_Test_1280_1300_DMI={'Nx': 142, 'alpha': 0.9393399620191372, 
        'density_W': 0.21694984991004806, 'sp': 0.4385711922163732, 'input_scaling': 0.10083027361212346, 
        'bias_scaling': -0.0774999653454595, 'reservoir_scaling': 0.32616628454389773, 'regularization': 0.006328620195770262}

        self.best_parameters_Low_Train_1200_1240_Test_1280_1300_DMI={'Nx': 259, 'alpha': 0.9824141990029831, 
        'density_W': 0.607443304634424, 'sp': 0.4846326779448509, 'input_scaling': 0.10013231449415638, 
        'bias_scaling': -0.07299163164012754, 'reservoir_scaling': 1.2678116257165473, 'regularization': 0.009473981448735718}

        self.best_parameters_Low_Train_All_Test_1280_1300_ZW={'Nx': 395, 'alpha': 0.9804536269749975, 
        'density_W': 0.9180984731105082, 'sp': 0.5003976556166263, 'input_scaling': 0.2905765114695555, 
        'bias_scaling': 0.24136489696981353, 'reservoir_scaling': 0.894893705048099, 'regularization': 0.008885085664251634}

        self.best_parameters_Low_Train_1200_1240_Test_1280_1300_ZW_weights_train_no_validation={'Nx': 293, 'alpha': 0.9982299094169474, 
        'density_W': 0.9420140188575027, 'sp': 0.774441638858245, 'input_scaling': 0.184466297625158, 'bias_scaling': 1.6523436055522376, 
        'reservoir_scaling': 1.9838790754012894, 'regularization': 0.0082728138467087}

        self.best_parameters_Low_Train_1200_1240_Test_1280_1300_ZW_no_weights={'Nx': 916, 'alpha': 0.9699668946516335, 
        'density_W': 0.5231518181198435, 'sp': 0.60976073704212, 'input_scaling': 0.10010437746727804, 'bias_scaling': -1.9816996009479984, 
        'reservoir_scaling': 1.6000985639084426, 'regularization': 0.007140604350577291}

        self.best_parameters_Low_Train_1200_1240_Test_1280_1300_ZW_no_train_weights_validation={'Nx': 872, 'alpha': 0.9726448054407866, 
        'density_W': 0.8181759418859526, 'sp': 0.7821630985423463, 'input_scaling': 0.10029174613321763, 'bias_scaling': 1.222386779645628, 
        'reservoir_scaling': 1.580971568356014, 'regularization': 0.004567646680106391}

        self.best_parameters_Low_Train_1200_1240_Test_1280_1300_ZW_both_weights={'Nx': 575, 'alpha': 0.9997854372279796, 
        'density_W': 0.1368180512032827, 'sp': 0.658646081172918, 'input_scaling': 0.18416817520563442, 
        'bias_scaling': 0.3465791346289395, 'reservoir_scaling': 1.8834055777056848, 'regularization': 0.009823578024190578}

        self.best_parameters_Low_Train_All_Test_1280_1300_ZW_both_weights_new={'Nx': 649, 'alpha': 0.9996793671020994, 
        'density_W': 0.24843838281419248, 'sp': 0.7505146304532544, 'input_scaling': 0.13453542130694138, 
        'bias_scaling': 0.14677442982066966, 'reservoir_scaling': 1.919564090099747, 'regularization': 0.009517399739240909}

        #self.best_parameters_High_Train_All_Test_1280_1300_ZW_both_weights_new={'Nx': 48, 'alpha': 0.73907982807996, 
        #'density_W': 0.7894365160596869, 'sp': 0.2570857723840986, 'input_scaling': 0.1130619949274585, 'bias_scaling': -1.1785845924996268, 
        #'reservoir_scaling': 0.7176198536990697, 'regularization': 0.0035239712436885133}

        self.best_parameters_High_Train_All_Test_280_300_ZW_both_weights_new={'Nx': 6, 'alpha': 0.7577708636143243, 'density_W': 0.7345216732705784, 
        'sp': 0.7347993028689438, 'input_scaling': 0.19546883934831755, 'bias_scaling': 0.011917975567641403, 'reservoir_scaling': 0.4706015782100997, 
        'regularization': 0.003168985913865958}    

        self.best_parameters_High_Train_All_Test_280_300_ZW_both_weights_balanced={'Nx': 54, 'alpha': 0.9729633235036554, 
        'density_W': 0.6452700670032729, 'sp': 0.4216934090959337, 'input_scaling': 0.10192094691534526, 'bias_scaling': -0.26921362213051236, 
        'reservoir_scaling': 1.4938320173723325, 'regularization': 0.005612674304390568}

        self.best_parameters_High_Train_All_Test_280_300_ZW_both_weights_2_variables={'Nx': 8, 'alpha': 0.7075851587791707, 
        'density_W': 0.9265252945836739, 'sp': 0.16000174843073306, 'input_scaling': 0.10222342887648272, 
        'bias_scaling': -1.5901784010778965, 'reservoir_scaling': 1.9171411819955664, 'regularization': 0.006113408778743627}

        self.best_parameters_High_Train_All_Test_280_300_ZW_no_weights={'Nx': 5, 'alpha': 0.964018732183177, 
        'density_W': 0.9002279329920887, 'sp': 0.4820027765772496, 'input_scaling': 0.12701160048429339, 
        'bias_scaling': -0.2959991161660833, 'reservoir_scaling': 0.4610960371874648, 'regularization': 0.0026721763791360993}

        self.best_parameters_High_Train_All_Test_280_300_ZW_no_weights_2_variables={'Nx': 124, 'alpha': 0.98503012019514, 
        'density_W': 0.6586800075141145, 'sp': 0.4041472742958729, 'input_scaling': 0.10151455399513387, 
        'bias_scaling': -1.9384679835567402, 'reservoir_scaling': 0.2409847261085605, 'regularization': 0.009042064147149299}

        #self.best_parameters_High_Train_All_Test_280_300_ZW_no_weights_2_variables_cycle={'Nx': 85, 'alpha': 0.9372380070101163, 
        #'density_W': 0.7018147161197015, 'sp': 0.1807814517047378, 'input_scaling': 0.10039202258253252, 'bias_scaling': 0.11970088855121855, 
        #'reservoir_scaling': 0.40500150850485667, 'regularization': 0.009313954326387272}

        self.best_parameters_High_Train_All_Test_280_300_ZW_no_weights_3_variables_cycle={'Nx': 6, 'alpha': 0.3617705522009436, 
        'density_W': 0.6548333642527416, 'sp': 0.1909968562214259, 'input_scaling': 0.18858827988109045, 'bias_scaling': 1.824185781449707, 
        'reservoir_scaling': 1.8480150759217437, 'regularization': 0.007330654710437044}

        self.best_parameters_High_Train_All_Test_280_300_ZW_no_weights_3_variables_cycle={'Nx': 6, 'alpha': 0.961584741621449, 
        'density_W': 0.8455499448034939, 'sp': 0.9495510524669182, 'input_scaling': 0.17971181807135028, 'bias_scaling': -0.3381714708063203, 
        'reservoir_scaling': 0.5366022636287752, 'regularization': 0.006444115353185831}

        #self.best_parameters_High_Train_All_Test_280_300_ZW_no_weights_2_variables_cycle={'Nx': 35, 'alpha': 0.7768621476970227, 
        #'density_W': 0.9975175230193586, 'sp': 0.12019047116537061, 'input_scaling': 0.10042991422659675, 'bias_scaling': -1.7280195769845466, 
        #'reservoir_scaling': 0.7412230199536594, 'regularization': 0.009994689948966457}

        self.best_parameters_High_Train_All_Test_280_300_ZW_no_weights_2_variables_cycle={'Nx': 38, 'alpha': 0.9991841951453823, 
        'density_W': 0.7400632917704144, 'sp': 0.29083382297321453, 'input_scaling': 0.10106772380741191, 'bias_scaling': -0.24106447066304876, 
        'reservoir_scaling': 0.8934470965702411, 'regularization': 0.004874906880610721}

        self.best_parameters_High_Train_All_Test_280_300_ZW_both_weights_3_variables_cycle={'Nx': 53, 'alpha': 0.5112594907688404, 
        'density_W': 0.7720975024394954, 'sp': 0.33500734706994195, 'input_scaling': 0.10205763031234556, 'bias_scaling': -0.596494802975022, 
        'reservoir_scaling': 1.2841525956043416, 'regularization': 0.008580587035294752}

        self.best_parameters_High_Train_All_Test_280_300_ZW_both_weights_2_variables_cycle={'Nx': 33, 'alpha': 0.9540830855875259, 
        'density_W': 0.2599923431868164, 'sp': 0.4030960866340541, 'input_scaling': 0.10327654718870272, 'bias_scaling': 0.2635208812874231, 
        'reservoir_scaling': 0.17477311435433446, 'regularization': 0.00948957676187012}

        self.best_parameters_High_Train_All_Test_280_300_ZW_both_weights_2_variables_cycle_balanced={'Nx': 5, 'alpha': 0.9530553588135151, 
        'density_W': 0.4144203445414923, 'sp': 0.8590238921239689, 'input_scaling': 1.3237909403400623, 'bias_scaling': -0.7536808275732176, 
        'reservoir_scaling': 1.6431889211828492, 'regularization': 0.005695235233294711}

        self.best_parameters_High_Train_All_Test_280_300_ZW_both_weights_3_variables_cycle_balanced={'Nx': 26, 'alpha': 0.7760093075911738, 
        'density_W': 0.6317053806216838, 'sp': 0.2620383413538799, 'input_scaling': 0.10069423988922116, 'bias_scaling': 1.7580218402160313, 
        'reservoir_scaling': 0.2616423571077733, 'regularization': 0.007111727134019717}

        self.best_parameters_Real_test_2000_2020_lead3_cycle_weights={'Nx': 187, 'alpha': 0.9643814082261515, 'density_W': 0.5793819258786342, 
        'sp': 0.3931321979499606, 'input_scaling': 0.10199872325956419, 'bias_scaling': -0.21106590576858095, 'reservoir_scaling': 1.110243827880433, 
        'regularization': 0.009485625235562822, 'training_weights': 0.8634011025320238}

        self.best_parameters_Real_test_2000_2020_lead6_cycle_weights={'Nx': 76, 'alpha': 0.7973917942876632, 'density_W': 0.890191741730807, 
        'sp': 0.14942816928012534, 'input_scaling': 0.10160554121712712, 'bias_scaling': -1.7803724614701637, 'regularization': 0.005378128683267119, 
        'training_weights': 0.6278948940224501}

        self.best_parameters_Real_test_2000_2020_lead6_cycle_no_weights={'Nx': 204, 'alpha': 0.9831406057445076, 'density_W': 0.9816977972603693, 
        'sp': 0.16291643612149118, 'input_scaling': 0.10083451334325941, 'bias_scaling': 0.20263264335260212, 'reservoir_scaling': 1.9531472171723576, 
        'regularization': 0.006091819431578127, 'training_weights': 1}

        self.best_parameters_Real_test_2000_2020_lead9_cycle_weights={'Nx': 104, 'alpha': 0.9355860934991769, 'density_W': 0.9993827139367203,
        'sp': 0.6517690881979338, 'input_scaling': 0.10238521656101555, 'bias_scaling': -1.6510358276502208, 'regularization': 0.007991273154963821,
        'training_weights': 0.9746645028818633}

        self.best_parameters_Real_test_2000_2020_all_leads_cycle_weights={'Nx': 89, 'alpha': 0.9784727131711992, 'density_W': 0.8001765772195408, 
        'sp': 0.2038530417092747, 'input_scaling': 0.1005263523305135, 'bias_scaling': 0.2757342478028685, 'regularization': 0.0077226333084101815, 
        'training_weights': 0.504943212033266}

        self.best_parameters_Real_test_2000_2020_lead3_no_cycle_weights={'Nx': 760, 'alpha': 0.9840624912823496, 'density_W': 0.8671021935344553, 
        'sp': 0.42592881007618105, 'input_scaling': 0.10104509389120499, 'bias_scaling': -0.2088919761956143, 'regularization': 0.005618988223609215, 
        'training_weights': 0.7790128528930409}

        self.best_parameters_Real_test_2000_2020_lead6_no_cycle_weights={'Nx': 920, 'alpha': 0.9081967497190402, 'density_W': 0.9985126453765443, 'sp': 0.29712989402014656, 
        'input_scaling': 0.1672983759202782, 'bias_scaling': -0.24755823854987377, 'regularization': 0.006903296758232935, 
        'training_weights': 0.6197155932854836}

        self.best_parameters_Real_test_2000_2020_lead9_no_cycle_weights={'Nx': 175, 'alpha': 0.9989377921380705, 'density_W': 0.8279709117549123, 
        'sp': 0.41943069991733783, 'input_scaling': 0.1065602472931754, 'bias_scaling': 1.9141039570202265, 'regularization': 0.00783929730161658, 
        'training_weights': 0.500599989756895}

        self.best_parameters_Real_test_2000_2020_lead9_cycle_no_weights={'Nx': 27, 'alpha': 0.8713610029668312, 'density_W': 0.7435638617860093, 
        'sp': 0.1601090742893871, 'input_scaling': 0.10074257119453403, 
        'bias_scaling': 0.7951474619851122, 'regularization': 0.008698243327090313,'training_weights': 1}

        self.best_parameters_Real_test_2000_2020_3_variables_lead3_cycle_weights={'Nx': 278, 'alpha': 0.7133515403066817, 'density_W': 0.7183142351261314, 
        'sp': 0.580262798908616, 'input_scaling': 0.10016049677508085, 'bias_scaling': 8.523090868064237, 'regularization': 0.007919590933992372, 
        'training_weights': 0.6013222636477142}

        self.best_parameters_Real_test_2000_2020_3_variables_lead6_cycle_weights={'Nx': 166, 'alpha': 0.7028006164347054, 'density_W': 0.594655164613185, 
        'sp': 0.43222497631541185, 'input_scaling': 0.10016666898237274, 'bias_scaling': -4.234946641817415, 'regularization': 0.00972907129369673, 
        'training_weights': 0.8982607682049991}

        self.best_parameters_Real_test_2000_2020_3_variables_lead9_cycle_weights={'Nx': 151, 'alpha': 0.5341897024755526, 'density_W': 0.9982617481202998, 
        'sp': 0.1344924549638502, 'input_scaling': 0.10184914188478504, 'bias_scaling': 8.715244002769172, 'regularization': 0.00881387525622452, 
        'training_weights': 0.7509939098014848}


        self.best_parameters_Low_Train_All_Test_1280_1300_2_variables_weights={'Nx': 936, 'alpha': 0.9538765272927417, 'density_W': 0.1249525997834422, 
        'sp': 0.7753961636699088, 'input_scaling': 0.10044745333965205, 'bias_scaling': 0.8535593694183383, 'regularization': 0.0032408107148349783, 
        'training_weights': 0.7502628138210912}

        self.best_parameters_Low_Train_All_Test_1280_1300_2_variables_no_weights={'Nx': 457, 
        'alpha': 0.9820519387379981, 'density_W': 0.3935599466544897, 'sp': 0.6491167143733372, 
        'input_scaling': 0.10096679373793803, 'bias_scaling': 1.3118525389650513, 
        'regularization': 0.006905634959884313,'training_weights':1}

        self.best_parameters_Low_Train_All_Test_1280_1300_3_variables_weights={'Nx': 833, 'alpha': 0.9622839007361127, 'density_W': 0.15715228808779944, 
        'sp': 0.6274778681600724, 'input_scaling': 0.10283480842840215, 'bias_scaling': 0.8298971633970523, 
        'regularization': 0.002588394667394362, 'training_weights': 0.7416451826504614}

        self.best_parameters_Low_Train_All_Test_1280_1300_3_variables_no_weights={'Nx': 311, 
        'alpha': 0.9266111836020712, 'density_W': 0.8453579048432875, 'sp': 0.670875562955639, 
        'input_scaling': 0.10035936395664548, 'bias_scaling': 1.7989137748322923, 
        'regularization': 0.002694687089726066,'training_weights': 1}


        self.best_parameters_Real_test_2000_2020_3_variables_cycle_weights_lead_6={'Nx': 170, 
        'alpha': 0.6167616319239702, 'density_W': 0.3906828379400824, 'sp': 0.39481992172398656, 
        'input_scaling': 0.12034461640168091, 'bias_scaling': -9.93030460006394, 
        'regularization': 0.008442092017791985, 'training_weights': 0.8905775726154873}

        self.best_parameters_Real_test_2000_2020_3_variables_cycle_weights_lead9={'Nx': 70, 
        'alpha': 0.8712719517192573, 'density_W': 0.7333140530870746, 'sp': 0.4406918119732519, 
        'input_scaling': 0.10038598680819974, 'bias_scaling': -0.229423203892988, 
        'regularization': 0.005932713801547058, 'training_weights': 0.9809709616840127}

        self.best_parameters_Real_test_2000_2020_3_variables_cycle_weights_lead3={'Nx': 362, 
        'alpha': 0.9763264067560792, 'density_W': 0.46286774957723903, 'sp': 0.9174914604997698, 
        'input_scaling': 0.10035167643214607, 'bias_scaling': 8.698779407668082, 
        'regularization': 0.009434073051810725, 'training_weights': 0.7458036368855677}

        self.best_parameters_Real_test_2000_2020_3_variables_cycle_weights_lead3_3_month={'Nx': 68, 
        'alpha': 0.7167323471516065,'density_W': 0.5044140524317876, 'sp': 0.42413621719793465, 
        'input_scaling': 0.5791941993484875, 'bias_scaling': -7.051908216144154, 
        'regularization': 0.008226502614251775, 'training_weights': 0.8968251356438729}


        self.best_parameters_basic_bifurcation={'Nx': 43, 'alpha': 0.8155607242479126, 
        'density_W': 0.775296670421177, 'sp': 0.1741024228202857, 'input_scaling': 0.8533348064638967, 
        'bias_scaling': -0.18754064687938815, 'reservoir_scaling': 1.7982702107493431, 
        'regularization': 0.008871783641979608, 'training_weights': 1}


        self.RealLead3={'Nx': 119, 'alpha': 0.9962087706351727, 'density_W': 0.6767159499174341, 
        'sp': 0.4755089666421766, 'input_scaling': 0.1006561718433378, 'bias_scaling': -5.0381468748650855, 
        'regularization': 0.008587281892691763, 'training_weights': 0.913774640421957}

        self.best_parameters_Real_test_2000_2020_3_variables_cycle_weights_lead6_3_month={'Nx': 68,
         'alpha': 0.6712879965773343, 'density_W': 0.24999045644565807, 'sp': 0.16234423658691402, 
         'input_scaling': 0.10069244873518965, 'bias_scaling': 6.448528709357617, 
         'regularization': 0.009290063534830926, 'training_weights': 0.53760966072984}

        
        self.best_parameters_Real_test_2000_2020_3_variables_cycle_weights_lead9_3_month={'Nx': 88, 
        'alpha': 0.6901927356765833, 'density_W': 0.3184019792710572, 'sp': 0.39819133116824623, 
        'input_scaling': 0.10164629911714049, 'bias_scaling': -4.595478796212109, 
        'regularization': 0.008588340243463869, 'training_weights': 0.9381613916252348}

        self.best_parameters_Real_test_2000_2020_3_variables_cycle_no_weights_lead6_3_month={'Nx': 109, 
        'alpha': 0.7662305648930938, 'density_W': 0.7570131189616784, 'sp': 0.35180974879076976, 
        'input_scaling': 0.10082970715403994, 'bias_scaling': -9.172140124130625, 
        'regularization': 0.00443663173401324,'training_weights': 1}

        self.best_parameters_Real_test_2000_2020_3_variables_cycle_no_weights_lead9_3_month={'Nx': 99, 
        'alpha': 0.7892585546661219, 'density_W': 0.5484376191865031, 'sp': 0.8157274147150031, 
        'input_scaling': 0.10024500011761879, 'bias_scaling': -6.118521319135327, 
        'regularization': 0.008941384901294818, 'training_weights':1}

        self.best_parameters_Real_test_2000_2020_3_variables_cycle_no_weights_lead3_3_month={'Nx': 95, 
        'alpha': 0.95266265024487, 'density_W': 0.9722431092125333, 'sp': 0.903224999946844, 
        'input_scaling': 0.10008625286168459, 'bias_scaling': -5.844007685711016, 
        'regularization': 0.006619753643880574, 'training_weights':1}

        self.best_parameters_Real_test_2000_2020_3_variables_no_cycle_weights_lead3_3_month={'Nx': 399, 
        'alpha': 0.9888246912431763, 'density_W': 0.42847477384903765, 'sp': 0.9513664799948958, 
        'input_scaling': 0.10093950394246953, 'bias_scaling': 2.4848850515470264, 
        'regularization': 0.009014725912488123, 'training_weights': 0.8826253306244725}
        


        self.best_parameters_Real_test_2000_2020_3_variables_no_cycle_weights_lead6_3_month={'Nx': 322, 
        'alpha': 0.5280586864898207, 'density_W': 0.7292836180975341, 'sp': 0.7121969558617804, 
        'input_scaling': 0.10072615162281053, 'bias_scaling': 5.521163314791265, 
        'regularization': 0.009136904730137554, 'training_weights': 0.8155151796063381}

        self.best_parameters_Real_test_2000_2020_3_variables_no_cycle_weights_lead9_3_month={'Nx': 205, 
        'alpha': 0.8873218512091126, 'density_W': 0.7674902706556228, 'sp': 0.4608854106578766,
         'input_scaling': 0.10053804824740191, 'bias_scaling': 3.4300865710090065, 
         'regularization': 0.007667142546956263, 'training_weights': 0.8310506147763175}


        self.best_parameters_Real_test_2000_2020_3_variables_no_cycle_no_weights_lead3_3_month={'Nx': 434, 
        'alpha': 0.6077202448034815, 'density_W': 0.5079502112053407, 'sp': 0.4789112238728968, 
        'input_scaling': 0.10047477129221354, 'bias_scaling': -3.2699897311718216, 
        'regularization': 0.009596017323896805, 'training_weights': 1}

        self.best_parameters_Real_test_2000_2020_3_variables_no_cycle_no_weights_lead6_3_month={'Nx': 101, 
        'alpha': 0.575790015566193, 'density_W': 0.9448987320093662, 'sp': 0.7258051871305057, 
        'input_scaling': 0.17245277972673845, 'bias_scaling': 3.4549735710509877, 
        'regularization': 0.006819549596044436,'training_weights': 1}

        self.best_parameters_Real_test_2000_2020_3_variables_no_cycle_no_weights_lead9_3_month={'Nx': 90, 
        'alpha': 0.7331352492594764, 'density_W': 0.8931597389058104, 'sp': 0.11867034433961905, 
        'input_scaling': 0.10113517068393633, 'bias_scaling': 8.064305639078349, 
        'regularization': 0.0066343865189604015, 'training_weights': 1}

        
        self.best_parameters_Real_test_2000_2020_3_variables_no_cycle_weights_leadAll_3_month={'Nx': 190, 
        'alpha': 0.8208638028308965, 'density_W': 0.44517884267781727, 'sp': 0.4057221557564086, 
        'input_scaling': 0.10066269090042744, 'bias_scaling': 0.2717142860888946, 
        'regularization': 0.005927095750490812, 'training_weights': 0.9337835500294248}
        
