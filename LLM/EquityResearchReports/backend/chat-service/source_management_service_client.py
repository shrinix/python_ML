#This is a test client for source-management-service.
#It sends a POST request to create a new source.
#Then it sends a GET request to get all sources.
#Then it sends a GET request to get the source with the id 1.
#Then it sends a PUT request to update the source with the id 1.
#Then it sends a DELETE request to delete the source with the id 1.
#
#To run this script, you need to have source-management-service running.
#You can run it using the following command:
#python source-management-service.py


import requests

url = 'http://localhost:5003/source'

headers = {
    # 'Authorization': 'Bearer YOUR_ACCESS_TOKEN',
    'Content-Type': 'application/json'
}

# Create a new source entry
def create_source_entry(company_name, pdf, status="inactive"):
    data = {
        'company_name': company_name,
        'pdf': pdf,
        'status': status
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200 and response.headers.get('Content-Type') == 'application/json':
        print(response.json())
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")

# Get all source entries
def get_source_entries():
    response = requests.get(url, headers=headers)
    if response.status_code == 200 and response.headers.get('Content-Type') == 'application/json':
        print(response.json())
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")

# Get the source entry with the id 1
def get_source_entry(id):
    response = requests.get(url + '/'+id, headers=headers)
    if response.status_code == 200 and response.headers.get('Content-Type') == 'application/json':
        print(response.json())
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")
#Get source entry by name in order to get its id
def get_source_entry_by_name(company_name):
    response = requests.get(url, headers=headers)
    if response.status_code == 200 and response.headers.get('Content-Type') == 'application/json':
        sources = response.json()
        for source in sources:
            if source['company_name'] == company_name:
                return source['id']
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")
    return None
#get all source entries with a specific status
def get_source_entries_by_status(status):
    sources = []
    response = requests.get(url, headers=headers)
    if response.status_code == 200 and response.headers.get('Content-Type') == 'application/json':
        sources = response.json()
        #filter sources by status
        sources = [source for source in sources if source['status'] == status]
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")

    return sources
#get all active source entries
def get_active_source_entries():
    sources = []
    active_url= url + '/active'
    response = requests.get(active_url, headers=headers)
    if response.status_code == 200 and response.headers.get('Content-Type') == 'application/json':
        sources = response.json()
        #filter sources by status
        sources = [source for source in sources if source['status'] == 'active']
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")

    return sources
# Update the source entry with the id 1
def update_source_entry(id, company_name, pdf, status="inactive"):
    data = {
        'company_name': company_name,
        'pdf': pdf,
        'status': status
    }
    response = requests.put(url + '/'+str(id), json=data, headers=headers)
    if response.status_code == 200 and response.headers.get('Content-Type') == 'application/json':
        print(response.json())
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")

# Delete the source with the id 1
def delete_source_entry(id):
    response = requests.delete(url + '/'+id, headers=headers)
    if response.status_code == 200 and response.headers.get('Content-Type') == 'application/json':
        print(response.json())
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")

#Delete all sources
def delete_all():
    response = requests.delete(url, headers=headers)
    if response.status_code == 200 and response.headers.get('Content-Type') == 'application/json':
        print(response.json())
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")


if __name__ == '__main__':
    # Create a new source entries
    # create_source_entry("Apple Inc", "apple-inc-2010-goldman-sachs.pdf")
    # create_source_entry("3P Learning", "3p-learning-2015-db.pdf")
    # create_source_entry("ABB", "abb-2015-nomura-global-markets-research.pdf")
    # create_source_entry("CBS Corporation", "cbs-corporation-2015-db.pdf")
    # create_source_entry("Duke Energy", "duke-energy-2015-gs-credit-research.pdf")
    # create_source_entry("Imperial Oil Limited", "imperial-oil-limited-2013-rbc-capital-markets.pdf")
    # create_source_entry("Premier Foods", "premier-foods-2015-bc-credit-research.pdf")
    # create_source_entry("Sanofi", "sanofi-2014-gs-credit-research.pdf")
    # create_source_entry("Schneider Electric", "schneider-electric-2015-no.pdf")
    # create_source_entry("The Walt Disney Company", "the-walt-disney-company-2015-db.pdf")
    # create_source_entry("Virgin Money Holdings", "virgin-money-holdings-2015-gs.pdf")

    # id = get_source_entry_by_name("3P Learning")
    # #update 3P Learning source to active status
    # update_source_entry(id, "3P Learning", "3p-learning-2015-db.pdf", "active")    

    # #Get all sources
    # get_source_entries()

    # active_entries = get_source_entries_by_status("active")
    active_entries = get_active_source_entries()
    print(active_entries)