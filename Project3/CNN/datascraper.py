import csv

# Open the original CSV file and create a reader to read its contents
with open('Data_Entry.csv', 'r') as original_file:
    reader = csv.reader(original_file)
    
    # Create a new CSV file to store the filtered data
    with open('Filtered_dataset.csv', 'w') as filtered_file:
        writer = csv.writer(filtered_file)
        
        # Loop through each row in the original CSV file
        for row in reader:
            # Check if the follow-up column contains a non-zero value
            if row[2] == '0':
                # If it does, write the row to the filtered CSV file
                # but only include the Image Index, Finding Labels, and Follow-up columns
                writer.writerow([row[0], row[1], row[2]])
