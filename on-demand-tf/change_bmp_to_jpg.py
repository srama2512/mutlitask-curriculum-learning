import os
import subprocess
from tqdm import tqdm

classNames = [ l.strip()[1:] for l in open(os.path.join(data_dir,'ClassName.txt')).readlines() ]
files = [(os.path.join(data_dir,c,f),i)
            for i,c in enumerate(classNames)
            for f in os.listdir(os.path.join(data_dir,c))
                if f.endswith(".jpg")]
fname = tf.placeholder(tf.string)
binary = tf.read_file(fname)
image = tf.image.decode_jpeg(binary,channels=3)

sess = tf.Session()
for path,l in tqdm(files):
    try :
        _ = sess.run(image,feed_dict={fname:path})
    except Exception as e:
        tqdm.write(path)
        print(e)

wrong_files = [
#'datasets/SUN397/g/gazebo/exterior/sun_akkhccpioqknevsw.jpg',
#'datasets/SUN397/i/islet/sun_ahobdrltxklfrjcp.jpg',
#'datasets/SUN397/s/server_room/sun_bvxousqntviqsdqq.jpg',
#'datasets/SUN397/c/classroom/sun_aiouiophdojjljpk.jpg',
#'datasets/SUN397/w/windmill/sun_bnvknudhvijnrlew.jpg',
#'datasets/SUN397/v/vineyard/sun_blvwshpgrpqramln.jpg',
#'datasets/SUN397/c/clean_room/sun_aukncbfrsjokobzo.jpg',
#'datasets/SUN397/a/abbey/sun_aqshzhndxgbjzfhn.jpg',
#'datasets/SUN397/p/pantry/sun_bewvkdqbsmllxcqf.jpg',
#'datasets/SUN397/b/baseball_field/sun_blcpbxhmsofmrdfl.jpg',
#'datasets/SUN397/c/creek/sun_bqgwyhijpzhpdqxg.jpg',
#'datasets/SUN397/t/tennis_court/outdoor/sun_bigyskuhsaormaeg.jpg',
#'datasets/SUN397/p/plaza/sun_bmawsbaitdpqnxmv.jpg',
#'datasets/SUN397/o/office_building/sun_bdjszqncotkyqiyr.jpg',
#'datasets/SUN397/t/theater/indoor_procenium/sun_axqhugfhwaicngwf.jpg',
#'datasets/SUN397/i/islet/sun_axyoqwlimchroili.jpg',
#'datasets/SUN397/a/apartment_building/outdoor/sun_acuzgrggofkmatpi.jpg',
#'datasets/SUN397/c/corral/sun_bjwnburtukssxfob.jpg',
#'datasets/SUN397/l/limousine_interior/sun_aajxcrzmjnuaoltv.jpg',
#'datasets/SUN397/p/patio/sun_bwmezvxdirlvahsi.jpg',
#'datasets/SUN397/w/waiting_room/sun_aefgqtpdweylvkwg.jpg',
#'datasets/SUN397/b/bazaar/indoor/sun_bgqptvebrkgytbfi.jpg',
#'datasets/SUN397/s/skyscraper/sun_bkmqdfrgxxzcigru.jpg',
#'datasets/SUN397/k/kindergarden_classroom/sun_aberbfmfrnqexldp.jpg',
#'datasets/SUN397/l/laundromat/sun_apwwjlkltuoymnio.jpg',
#'datasets/SUN397/w/waiting_room/sun_bicyxgrjdvyrslnb.jpg',
#'datasets/SUN397/b/bullring/sun_ahyiurfcycttetnj.jpg',
#'datasets/SUN397/s/swimming_pool/indoor/sun_btedcjfyzxonsttw.jpg',
#'datasets/SUN397/c/cubicle/office/sun_axwbqpyxfyevyyax.jpg',
#'datasets/SUN397/s/stadium/baseball/sun_bjwzfkcpdcmwkwvd.jpg',
#'datasets/SUN397/c/childs_room/sun_akvtrgxoikdpfltn.jpg',
#'datasets/SUN397/m/museum/indoor/sun_brpoxjopmnslgnjv.jpg',
#'datasets/SUN397/v/vineyard/sun_bqmlhayzezizport.jpg',
#'datasets/SUN397/w/warehouse/indoor/sun_bkmhhbdvfycbajvn.jpg',
#'datasets/SUN397/b/booth/indoor/sun_bngnicabuafdfizk.jpg',
#'datasets/SUN397/c/church/outdoor/sun_bhenjvsvrtumjuri.jpg',
#'datasets/SUN397/t/tree_house/sun_bjiwldemlzjhfkhk.jpg',
#'datasets/SUN397/m/mountain_snowy/sun_bffbbkghqwtgqmip.jpg',
#'datasets/SUN397/l/lobby/sun_bevdbeymsvlktnbx.jpg',
#'datasets/SUN397/r/restaurant_patio/sun_avipwboynqdgmmzh.jpg',
#'datasets/SUN397/p/playground/sun_blqgamrntajljtma.jpg',
#'datasets/SUN397/k/kitchen/sun_akndyxefbqzbqsqy.jpg',
#'datasets/SUN397/g/golf_course/sun_ajjvgigjtyjuqxub.jpg',
#'datasets/SUN397/c/conference_room/sun_bpyenksfiblxdbub.jpg',
#'datasets/SUN397/b/boardwalk/sun_brkucyfznnwbhryz.jpg',
#'datasets/SUN397/c/carrousel/sun_auwcbmlvbwixdzom.jpg',
#'datasets/SUN397/b/badminton_court/indoor/sun_alqkzsaxauesgwir.jpg',
#'datasets/SUN397/g/golf_course/sun_asbqjxocgmlenazd.jpg',
#'datasets/SUN397/v/vineyard/sun_bhwnukbgnueyvgau.jpg',
#'datasets/SUN397/w/warehouse/indoor/sun_blpphrffcwkelado.jpg',
#'datasets/SUN397/m/mountain/sun_byhztyvzwhsmkkhh.jpg',
#'datasets/SUN397/l/library/indoor/sun_bjznziofrhznhygb.jpg',
#'datasets/SUN397/a/auditorium/sun_adqmjrhfjyuzyvdd.jpg',
#'datasets/SUN397/i/islet/sun_anwfjifvecriqcwa.jpg',
#'datasets/SUN397/g/greenhouse/indoor/sun_bdyswncyawnxqunj.jpg',
]


for f in wrong_files:
    name, _ = os.path.splitext(f)
    os.rename(f, name+'.bmp')

data_dir= 'datasets/SUN397'
classNames = [ l.strip()[1:] for l in open(os.path.join(data_dir,'ClassName.txt')).readlines() ]

for c in classNames:
    script='/usr/bin/mogrify -format jpg %s/*.bmp -quality 100'%(os.path.join(data_dir,c))
    #print(script)
    try:
        subprocess.call([script],shell=True)
    except:
        pass
