

class Entities:
    def __init__(self) -> None:
        pass
    
    def main(self):
        return {
                "id": 0,
                "user": "http://localhost/users/4/",
                "frame": "http://localhost/frames/4000/",
                "detected_objects": []
                }
    
    def detectedObject(self):
        return {
                "cls": "http://localhost/classes/1/",
                "landing_status": "-1",
                "top_left_x": 262.87,
                "top_left_y": 734.47,
                "bottom_right_x": 405.2,
                "bottom_right_y": 847.3
                }