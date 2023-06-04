from toJson.entities import Entities


def jsonResult(boxes_list, labels_list, pist_score_list, id_num, frame):
    entities = Entities()
    detected_objects = []
    for box, label, pist_score in zip(boxes_list, labels_list, pist_score_list):
        detected_dict = entities.detectedObject()

        detected_dict["cls"] = f"http://localhost/classes/{str(label)}/"
        detected_dict["landing_status"] = str(pist_score)
        detected_dict["top_left_x"] = round(float(box[0]),2) if box[0]>0 else 0.
        detected_dict["top_left_y"] = round(float(box[1]),2) if box[1]>0 else 0.
        detected_dict["bottom_right_x"] = round(float(box[2]),2) if box[2]>0 else 0.
        detected_dict["bottom_right_y"] = round(float(box[3]),2) if box[3]>0 else 0.

        detected_objects.append(detected_dict)

    main_dict = entities.main()
    main_dict["id"] = int(id_num)
    main_dict["user"] = "http://localhost/users/4/"
    main_dict["frame"] = str(frame)
    main_dict["detected_objects"] = detected_objects

    return [main_dict]
