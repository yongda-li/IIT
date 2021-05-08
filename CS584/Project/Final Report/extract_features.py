def extract_features(image_id, model):
    file = BASE_DIR/'train_images'/image_id
   
    img = load_img(file, target_size=(224,224))
   
    img = np.array(img) 
    
    reshaped_img = img.reshape(1,224,224,3) 
    
    imgx = preprocess_input(reshaped_img)
    
    features = model.predict(imgx, use_multiprocessing=True)
    
    return features