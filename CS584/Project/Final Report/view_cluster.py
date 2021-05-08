def view_cluster(cluster):
    plt.figure(figsize = (25,25));
    
    files = groups[cluster]
    
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 25")
        start = np.random.randint(0,len(files))
        files = files[start:start+25]
    
    for index, file in enumerate(files):
        plt.subplot(5,5,index+1);
        img = load_img(BASE_DIR/'train_images'/file)
        img = np.array(img)
        plt.imshow(img)
        plt.title(file)
        plt.axis('off')