# Mapping for categorical values

# Sorted region mapping
region_mapping = {
    region: index for index, region in enumerate(sorted([
        'Andheri West', 'Naigaon East', 'Borivali West', 'Panvel', 'Mira Road East',
        'Parel', 'Boisar', 'Santacruz East', 'Badlapur East', 'Fort', 'Badlapur West',
        'Khopoli', 'Chembur', 'Jogeshwari West', 'Vasai', 'Kalamboli', 'Powai',
        'Ghansoli', 'Thane West', 'Vikhroli', 'Bhiwandi', 'Airoli', 'Ambernath West',
        'Borivali East', 'Virar', 'Kharghar', 'Ulwe', 'Kamothe', 'Jogeshwari East',
        'Mulund West', 'Palghar', 'Goregaon West', 'Taloja', 'Rasayani',
        'Ghatkopar East', 'Ulhasnagar', 'Dombivali', 'Kewale', 'Nala Sopara',
        'Goregaon East', 'Kandivali East', 'Kurla', 'Andheri East', 'Dahisar',
        'Karanjade', 'Mahim', 'Vashi', 'Malad West', 'Girgaon', 'Dadar West',
        'Bandra West', 'Kandivali West', 'Kalyan West', 'Neral', 'Kalyan East',
        'Karjat', 'Ghatkopar West', 'Dronagiri', 'Mulund East', 'Navade', 'Ambivali',
        'Agripada', 'Owale', 'Ville Parle East', 'Kalwa', 'Khar', 'Santacruz West',
        'Nerul', 'Kanjurmarg', 'Vangani', 'Bhayandar East', 'Seawoods', 'Sewri',
        'Ambernath East', 'Nilje Gaon', 'Prabhadevi', 'Matunga', 'Lower Parel',
        'Titwala', 'Shil Phata', 'Koper Khairane', 'Napeansea Road', 'Bhandup West',
        'Koproli', 'Anjurdive', 'Wadala', 'Sion', 'Taloje', 'Cuffe Parade',
        'Bhandup East', 'Byculla', 'Tardeo', 'Vasai West', 'Vasai East', 'Malad East',
        'Colaba', 'Thane East', 'Nalasopara East', 'Deonar', 'Nahur East', 'Sanpada',
        'Sector 21 Kamothe', 'Saphale', 'Kasheli', 'Juinagar', 'Worli', 'Panch Pakhdi',
        'Mazagaon', 'Hiranandani Estates', 'Belapur', 'Vichumbe', 'Bandra East',
        'Sector 17 Ulwe', 'Sector 23 Ulwe', 'Nalasopara West', 'Mahalaxmi',
        'Sector 20 Kamothe', 'Dadar East', 'Sector 19 Kamothe', 'Shahapur',
        'Sector 30 Kharghar', 'Asangaon', 'Vikroli East', 'Mira Road',
        'Kanjurmarg East', 'Rambaug', 'Sector-12 Kamothe', 'Juhu', 'Ville Parle West',
        'Mazgaon', 'Virar East', 'Khar West', 'Sector 8 New Panvel', 'Rabale',
        'Bhayandar West', 'Sector 20 Ulwe', 'Sector 22 Kamothe', 'Sector 21 Nerul',
        'Bandra Kurla Complex', 'Sector 14 Vashi', 'Mumbai Central', 'Virar West',
        'Sector 11 Koparkhairane', 'Unnat Nagar', 'Diva', 'Palava', 'Dombivali East',
        'Sector-14 Koparkhairane', 'Greater Khanda', 'Sector-35D Kharghar', 'Umroli',
        'Sector-9 Ulwe', 'Govandi', 'Vile Parle West', 'Matunga West',
        'Sector-3 Ulwe', 'Kasaradavali Thane West', 'Kurla East', 'Pestom Sagar Colony',
        'Sector 12 Kharghar', 'Police Colony', 'Dahisar West', 'Marine Lines',
        'Sector 19 Kharghar', 'Kalher', 'Hindu Colony', 'Dahisar East',
        'Sector 9 Vashi', 'Khardi', 'Babulnath Road', 'Sector 21 Kharghar',
        'Dharavi', 'Vasind', 'Tilak Nagar', 'Ashok Nagar', 'Antop Hill',
        'Peddar Road', 'Kamathipura', 'Usarghar Gaon', 'Ambarnath', 'Patlipada',
        'Vevoor', 'Shelu', 'Kurla West', 'Goregaon', 'Naupada', 'Bhoiwada',
        'Sector 7 Kharghar', 'Roadpali', 'Sector-9 Kamothe', 'Borivali', 'Badlapur',
        'Khanda Colony', 'Dombivli (West)', 'GTB Nagar', 'Bandra', 'Kandivali',
        'Mahavir Nagar', 'Churchgate', 'Pali Hill', 'Manpada', 'Sector-50 Seawoods',
        'Uttan', 'Gauripada', 'Gandhar Nagar', 'Mahim West', 'Warai', 'Mumbai',
        'Khatiwali', 'Chandivali', 'Mumbra', 'Chembur East', 'Malabar Hill', 'Sector',
        'Uran', 'Manjarli', 'Ghodbunder Road', 'Mulund', 'Sector 18 Kharghar',
        'Palidevad', 'Juhu Scheme', 'Adaigaon', 'Versova', 'Sector-4 New Panvel',
        'Pen', 'Sector 6 Kamothe', 'Maneklal Estate', 'L I C Colony'
    ]))
}

# Age mapping
age_mapping = {
    'New': 1,
    'Resale': 2,
    'Unknown': 3
}

# Property type mapping
property_type_mapping = {
    'Studio Apartment': 1,
    'Apartment': 2,
    'Independent House': 3,
    'Penthouse': 4,
    'Villa': 5
}

# Status mapping
status_mapping = {
    'Ready to move': 1,
    'Under Construction': 2
}
