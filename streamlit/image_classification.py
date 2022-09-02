import keras
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from keras import optimizers
import tensorflow_addons as tfa

projection_dim = 64

def find_the_remedy(disease_code):
    diseases = {}
    keys = 38
    values = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']


    for i in range(0, keys):
    	diseases[i] = values[i]
        
    remedies = dict({
    	0:"Use Fungicide portion of an all-purpose fruit tree spray.", 
    	1:"If you must prune during the season, consider treating the sites with the antibiotic streptomycin or a copper-based fungicide.", 
    	2:"Choose resistant cultivars when available.", 
    	3:"NOT NEEDED ANY!", 
    	4:"NOT NEEDED ANY!" , 
    	5:"Most synthetic fungicides are preventative, not eradicative, so be pro-active about disease prevention. Maintain a consistent program from shuck fall through harvest.", 
    	6:"NOT NEEDED ANY!", 
    	7:"Fungicides may be needed to prevent significant loss when plants are infected early and environmental conditions favor disease.",
    	8:"Use resistant corn hybrids. Fungicides can also be beneficial, especially if applied early when few pustules have appeared on the leaves.", 
    	9:"Try using fungicides. For most home gardeners this step isn't needed, but if you have a bad infection, you may want to try this chemical treatment.", 
    	10:"NOT NEEDED ANY!", 
    	11:"Mancozeb, and Ziram are all highly effective against black rot. Because these fungicides are strictly protectants, they must be applied before the fungus infects or enters the plant.", 
    	12:"In addition to the fungicides labeled as pruning-wound protectants, consider using alternative materials, such as a wound sealant with 5% boric acid in acrylic paint (Tech-Gro B-Lock), which is effective against Eutypa dieback and Esca, or an essential oil (Safecoat VitiSeal).", 
    	13:"Plant less susceptible cultivars; application of Bordeaux mixture or other appropriate fungicide while vines are dormant may be necessary.", 
    	14:"NOT NEEDED ANY!", 
    	15:"There is no cure for citrus greening, which explains why spotting citrus greening disease symptoms early is so crucial– rapid removal of infected trees is the only way to stop the spread of the bacteria responsible.",
    	16:"Use copper, Oxytetracycline(Mycoshield and generic equivalents), and syllit plus captan.", 
    	17:"NOT NEEDED ANY!", 
    	18:"Use disease-free seed that has been produced in western states or seed that has been hot water treated.", 
    	19:"NOT NEEDED ANY!", 
    	20:"Try planting potato varieties that are resistant to the disease.", 
    	21:"Late blight is controlled by eliminating cull piles and volunteer potatoes, using proper harvesting and storage practices, and applying fungicides when necessary.", 
    	22:"NOT NEEDED ANY!", 
    	23:"NOT NEEDED ANY!",
    	24:"NOT NEEDED ANY!", 
    	25:"Use baking soda.", 
    	26:"Use resistant cultivars, fungicides.", 
    	27:"NOT NEEDED ANY!", 
    	28:"A plant with bacterial spot cannot be cured. Remove symptomatic plants from the field or greenhouse to prevent the spread of bacteria to healthy plants.", 
    	29:"The most common fungicides that are used for control of early blight include Mancozeb, Maneb, Chlorothalonil, difenoconazole and copper oxychloride.", 
    	30:"For the home gardener, fungicides that contain maneb, mancozeb, chlorothanolil, or fixed copper can help protect plants from late tomato blight. Repeated applications are necessary throughout the growing season as the disease can strike at any time.", 
    	31:"An apple-cider and vinegar mix is believed to treat the mold effectively.", 
    	32:"Use fungicides with active ingredients such as chlorothalonil, copper, or mancozeb will help reduce disease.", 
    	33:"Apply a pesticide specific to mites called a miticide.", 
    	34:"Products containing chlorothalonil, mancozeb, and copper oxychloride have been shown to provide good control of target spot in research trials.", 
    	35:"Use a neonicotinoid insecticide, such as dinotefuran (Venom) imidacloprid (AdmirePro, Alias, Nuprid, Widow, and others) or thiamethoxam (Platinum), as a soil application.", 
    	36:"Remove all infected plants and destroy them. Do NOT put them in the compost pile, as the virus may persist in infected plant matter.", 
    	37:"NOT NEEDED ANY!"
	})    

    disease = diseases.get(disease_code)
    remedy = remedies.get(disease_code)
    return disease,remedy


def plant_disease_detection(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 256, 256, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (256, 256)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255.0)
	normalized_image_array = tf.expand_dims(normalized_image_array, 0)
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability