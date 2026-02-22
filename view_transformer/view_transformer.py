import numpy as np 
import cv2

class ViewTransformer():
    def __init__(self, frame_width=1280, frame_height=720):
        court_width = 68
        court_length = 23.32

        # Reference resolution (1080p)
        ref_w = 1920
        ref_h = 1080
        
        # Original 1080p vertices for perspective transform
        ref_vertices = np.array([
            [110, 1035], 
            [265, 275], 
            [910, 260], 
            [1640, 915]
        ])
        
        # Scale vertices based on input frame dimensions
        scale_x = frame_width / ref_w
        scale_y = frame_height / ref_h
        
        self.pixel_vertices = ref_vertices.copy().astype(np.float32)
        self.pixel_vertices[:, 0] *= scale_x
        self.pixel_vertices[:, 1] *= scale_y
        self.pixel_vertices = self.pixel_vertices.astype(np.int32)
        
        self.target_vertices = np.array([
            [0,court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self,point):
        if point is None or len(point) < 2:
            return None
        try:
            p = (int(point[0]),int(point[1]))
            is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0 
            if not is_inside:
                return None

            reshaped_point = np.array(point).reshape(-1,1,2).astype(np.float32)
            transform_point = cv2.perspectiveTransform(reshaped_point,self.perspective_transformer)
            return transform_point.reshape(-1,2)
        except (ValueError, IndexError, TypeError):
            return None

    def add_transformed_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    if 'position_adjusted' not in track_info:
                        continue
                    position = track_info['position_adjusted']
                    if position is None:
                        tracks[object][frame_num][track_id]['position_transformed'] = None
                        continue
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed