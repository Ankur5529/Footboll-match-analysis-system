from sklearn.cluster import KMeans

import numpy as np


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}


    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Perform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        # Check if we have enough data (K-Means with 2 clusters requires at least 2 samples)
        if len(image_2d) < 2:
            return None
            
        kmeans.fit(image_2d)



        return kmeans

    def get_player_color(self,frame,bbox):
        y1, y2 = max(0, int(bbox[1])), max(0, int(bbox[3]))
        x1, x2 = max(0, int(bbox[0])), max(0, int(bbox[2]))
        image = frame[y1:y2, x1:x2]

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)
        if kmeans is None:
            return None

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self,frame, player_detections):
        
        if len(player_detections) == 0:
            # Default team colors if no players detected
            self.team_colors[1] = [255, 0, 0]  # Red
            self.team_colors[2] = [0, 0, 255]  # Blue
            return
        
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            if player_color is not None:
                player_colors.append(player_color)
        
        print(f"Collected {len(player_colors)} player colors for clustering.")
        if len(player_colors) > 0:
            print(f"Sample colors (up to 5): {player_colors[:5]}")
        
        if len(player_colors) < 2:
            # If less than 2 players, use default colors
            self.team_colors[1] = player_colors[0] if len(player_colors) > 0 else [255, 0, 0]
            self.team_colors[2] = [0, 0, 255]  # Default blue
            return
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        
        labels = kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Team cluster counts (Training Frame): {dict(zip(unique, counts))}")
        
        print(f"Team 1 Color: {self.team_colors[1]}")
        print(f"Team 2 Color: {self.team_colors[2]}")
        
        # Write debug info to file
        try:
            with open('clustering_debug.txt', 'w') as f:
                f.write(f"Collected {len(player_colors)} player colors for clustering.\n")
                if len(player_colors) > 0:
                     f.write(f"Sample colors: {player_colors[:5]}\n")
                f.write(f"Team cluster counts (Training Frame): {dict(zip(unique, counts))}\n")
                f.write(f"Team 1 Color: {self.team_colors[1]}\n")
                f.write(f"Team 2 Color: {self.team_colors[2]}\n")
        except Exception as e:
            print(f"Failed to write detailed debug info: {e}")
        



    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if not hasattr(self, 'kmeans') or self.kmeans is None:
            # Default to team 1 if kmeans not initialized
            self.player_team_dict[player_id] = 1
            return 1

        player_color = self.get_player_color(frame,player_bbox)
        if player_color is None:
            return 1

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1



        self.player_team_dict[player_id] = team_id

        return team_id
    


