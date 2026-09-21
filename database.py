import mysql.connector


def save_detection(
    object_name,
    confidence,
    track_id,
    image_path,
    x1,
    y1,
    x2,
    y2,
    camera_name
):

    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="object_detection_db"
    )

    cursor = connection.cursor()

    query = """
        INSERT INTO detections
        (
            object_name,
            confidence,
            track_id,
            image_path,
            x1,
            y1,
            x2,
            y2,
            camera_name
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    values = (
        object_name,
        confidence,
        track_id,
        image_path,
        x1,
        y1,
        x2,
        y2,
        camera_name
    )

    cursor.execute(query, values)

    connection.commit()

    cursor.close()
    connection.close()

    print("Detection saved successfully!")