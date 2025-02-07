import csv

def export_to_csv(data, filename='output.csv'):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Label', 'Text', 'Offset'])
            for dato in data:
                writer.writerow(dato)