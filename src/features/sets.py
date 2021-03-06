from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) | {'', '&', 'x', 'ft', 'lb', 'f'}

additional_syn = {'or': {'oil-rubbed', 'oil', 'rubbed'}}

blank = set([])

boolean_columns = {'Hooded', 'Handshower included', 'Hooks Included', 'Humidity Gauge', 'Elastic hem',
                   'Extra bulbs/fuses included', 'Cover Plate Included', 'Non marring', 'Nonconductive', 'Wood Veneer',
                   'Translucent fuel tank', 'Removes pollen', 'Tow hitch included', 'Dispenses Soap', 'Detects metal',
                   'Drawer storage', 'Retread', 'Textured Grip', 'Telescoping Handle', 'Weather Resistant',
                   'Extender Horn Included', 'Corded', 'Ice Crushing Button', 'Extendable', 'Flammable',
                   'Batteries Included', 'Short circuit protection', 'Attached scrubber', 'Gear Driven',
                   'Pet and livestock friendly', 'Fog Free', 'Blade storage', 'Strainer Included',
                   'Quick Connect Fan-Blade System', 'Built-in power outlets', 'Automatic', 'Transparent cover/window',
                   'Hearth included', 'Displays dew point', 'Mounting frame included', 'Trencher conversion capable',
                   'Sound insulation', 'Carrying case', 'Auto oiler', 'Electrostatic', 'Self-Locking',
                   'Seat can be added', 'Hose-End Timer', 'Etching capable', 'Electrical hazard protection',
                   'Corrosion Resistant', 'Removes odors', 'Adjustable Handle', 'Nonmarking sole', 'Cold applied',
                   'Novelty', 'Acoustical', 'Magnetized', 'Multiple heads included', 'Adjustable Thermostat',
                   'Solvent free', 'Includes drinkware', 'Cover Included', 'Out-of-Range Indicator', 'Conduit Fitting',
                   'Ionizing', 'Nondrip', 'Port Diverter', 'Machine Washable', 'Fire Retardant', 'Onboard cord storage',
                   'Adjustable shackle', 'Gravity feed hopper', 'Muffler', 'Engine oil included', 'Built In Grinder',
                   'Coring Tube Open/Close', 'Wire-Break Alarm', 'Tilt-in cleaning', 'Arched', 'Depth Adjustment',
                   'Adjustable hinges', 'AC adapter included', 'Paintable / Stainable', 'Replaceable point',
                   'Adjustable cutting depth', 'UV Protected', 'Extendable handle', 'Resealable Container',
                   'Latex based', 'Low profile', 'Trigger Lock', 'Brakes', 'Elastic banded waist',
                   'Built-in HEPA filter', 'Insect screen', 'Dump Cart', 'Holdbacks included', 'Removable Pulp',
                   'Adapters included', 'Warming Rack', 'Channel Volume Control', 'Fire rated', 'Watch Use',
                   'Stand Included', 'Low Battery Indicator', 'Mark Resistant', 'Applicator included', 'De-Icer',
                   'LCD display', 'Thumb Patch', 'Heat thermometer included', 'Lift Wire Included', 'Wet/Dry',
                   'Insulating', 'Airtight', 'Tiebacks/holdbacks included', 'UV lamp', 'Clean/Dirty Indicator',
                   'Touchpad Controls', 'Backboard/Rest/Rail', 'Weldable', 'Full leg wrap', 'Built-In Light',
                   'Hypoallergenic', 'Batteries Required', 'Self-diagnostic tools', 'Built-in inverter',
                   'Full reverse capability', 'Low battery indicator light', 'Staggered nail pattern', 'Rechargeable',
                   'Sleeve Included', 'Safe on Fabric', 'Cool Down Cycle', 'Bag Cutter', 'Riser Required',
                   'Outgoing mail slot/receptacle', 'Floatation Device', 'Replaceable Head', 'Sleep setting',
                   'Includes Software', 'Bagging Material Included', 'Lid', 'Shoulder harness/strap',
                   'Adjustable hanging length', 'Abrasion-Proof Display Lens', 'Lit', 'Lid Included',
                   'Lightning Protection', 'Sealed', 'Flexible', 'Protects against rust', 'Date display', 'Kick stand',
                   'Finials Included', 'No Lemon Policy', 'Interchangeable Tip?', 'Wall Mountable', 'Cord lock',
                   'Remote handset', 'Salt Resistant', 'Night Vision', 'On/Off Switch', 'Moisture Sealing',
                   'Automatic relocking', 'Matted', 'Premixed', 'Push to connect', 'Hidden Bake Element',
                   'Slip-resistant tub floor', 'D-Grip Handle', 'Swing-away', 'Fire and high heat resistant',
                   'Oil based', 'Demolition', 'Spark-resistant chain', 'Photo holder', 'Built-In CD Player', 'Seat',
                   'Fire Resistant', 'Westminster Bell', 'Transferrable', 'Dual voltage indication', 'Vacuum clean',
                   'Slip-resistant grip', 'Shower head Included', 'Telescoping', 'Integrated Wastebasket', 'Sandable',
                   'Window venting kit included', 'Rotary Blades', 'Built-in flange', 'Osha Required GFCI Outlets',
                   'Motion Sensing', 'Tool-less conversion', 'Shock spring', 'Skid shoes', 'LED backlight', 'Drillable',
                   'Removable tip', 'Float Switch', 'Travel/lightweight', 'Safety tread design',
                   'Automatic Emissivity Adjustment', 'Remote distance', 'Kevlar stitching', 'Movable fixtures',
                   'Removable Key Lock', 'Scratch Resistant', 'Blade Balancer Included', 'Sound',
                   'Displays wind direction and speed', 'Name card holders included', 'Left Side Inlet',
                   'Grill light included', 'Folding design', 'Key Fob Use', 'Reconditioned', 'Chilled Water',
                   'Gutter Tip', 'Tape Holder', 'Pedestal Model#', 'Accessories Included', 'Drop leaf', 'Parking Brake',
                   'Wall mount included', 'Pull Chain', 'Detects voltage', 'Two-sided design', 'Integral J-channel',
                   'Knife block included', 'Privacy glass', 'Nonslip base', 'Wildlife Guard', 'Master Volume Control',
                   'Primed', 'Flame Retardant', 'Shrink Resistant', 'Resettable circuit breaker',
                   'Reference Scale Included', 'Adjustable Cross Section', 'Rapid dissolving', 'Data hold',
                   'Convertable to Handheld', 'Irrigated', 'Self-closing valve', 'Slim line', 'Hydronic', 'Hammer loop',
                   'Pet Attachment', 'Self tapping', 'Battery Back-Up', 'Light Cover(s) Included',
                   'Color Changing Lights', 'Includes Play Deck or Tower', 'Battery tester included',
                   'Roller cover included', 'Screen Included', 'Cabinet storage', 'Zone-specific Sounds',
                   'Self-sealing', 'Waffle Maker', 'Anti-Bacterial or Disinfecting', 'Water resistant handrail',
                   'Box Spring Included', 'Rotisserie', 'Post and accessories included', 'Sap Groove',
                   'Fits over eyeglasses', 'Ventilated', 'Sensor bar', 'Can Be Used Below Grade', 'Mirror',
                   'Tinted glass', 'Data Log/Record', 'Microprocessor Controlled Sensor', 'Multiple time zones',
                   'Seating Bench', 'Corner unit', 'Dipole', 'Handles', 'Wrapped', 'Transport Caddy',
                   'Curtain Included', 'Photoluminescent', 'Ceiling use', 'Cushioned grip',
                   'Radiant/Underfloor Warming Approved', 'Humidity Sensing', 'Caddy/holder included',
                   'Extra Battery Included', 'Mantel included', 'Tower included', 'Cut to Length of Order', 'Threaded',
                   'Concealed Hardware', 'Outlet Cover', 'Auto Ranging', 'Accessories/Parts Included',
                   'Surge Protector', 'Coiled', 'Attached shelf', 'Hand brake', 'Pre-emergent',
                   'OSHA recommended safety latch', 'Pre-emergent weed control', 'Set', 'Suction Cup Bottom',
                   'Wireless Remote Included', 'No-rinse', 'Fixed Mount', 'Rotating', 'Non-Stick Surface',
                   'Power Washer Operation', 'Stackable', 'Replacement blade', 'Open Front', 'No-tool line replacement',
                   'Breakaway design', 'Letters/Numbers Included', 'Towable', 'Blower', 'Tie-down grommets',
                   'Side Inlet', 'Integrated tongue to frame construction', 'Gift Card Replacement',
                   'Packing Tape Included', 'Clean Basin Indicator', 'Push-to-Fit', 'Strainer Basket Included',
                   'Chalk included', 'Waterproof', 'Handle(s)', 'Interchangeable Tips Available', 'Kit', 'Moisturizing',
                   'Finished', 'Permanently flexible', 'Universal control', 'Chemical resistant',
                   'Accepts Credit Cards', 'Custom Door Kit Compatible', 'Direct Burial', 'Adjustable Forward Speed',
                   'Wheel Kit Included', 'Bag included', 'Slide-out', 'Padded seat', 'Attachment capable', 'Drainage',
                   'Sand included', 'Reinforced pockets', 'Biometric', 'UV Resistant', 'Built-in Wine Rack',
                   'In-Line Leaf Trap', 'Wand', 'Visual Alert', 'Sample', 'LowE rating', 'Front wheels', 'Dusk to Dawn',
                   'Scent', 'Batteries Included for Transmitter', 'Concrete Use', 'Door Alarm', 'Opaque',
                   'Hidden storage', 'Video Protection', 'Music Device Input', 'Attachments Included',
                   'Concealed Mounting Hardware', 'Adjustable Compartments', 'Dispenser Spout', 'Integral dish rack',
                   'Interchangeable Pins/Probes', 'Adjustable handle', 'Ergonomic Handle(s)', 'Releasable',
                   'Motion detecting', 'Lamp heads', 'Click locking', 'Dust Collection', 'Internal Lighting',
                   'Hour Meter', 'Hook included', 'Metal blade conversion kit', 'Detachable Canister',
                   'Pole extension included', 'Belled end', 'Installable over Cork Underlayment', 'Rust resistant deck',
                   'Beveled', 'Internet Enabled', 'Poultry shears', 'Pressure Rating', 'Light Bulbs Included',
                   'Toe Kicks Included', 'Dust Blower', 'Chain brake', 'Bulb(s) Included', 'Bluetooth',
                   'Multiple Songs', 'Clearing tool included', 'Cart Included', 'Torch included', 'Oven Convection',
                   'Adjustable Back Straps', 'Engine start function', 'Double Hinged', 'UL Fire Rated', 'Magnetic Tip',
                   'Vibration Alert', 'Stainless steel grind chamber', 'Bipole', 'Wet Location Use', 'Glued edge',
                   'Filter light reminder', 'Accepts Additional Cameras', 'Pedestal Included', 'Remote Zoom',
                   'Padding Attached', 'Drain Required', 'Extended Control', 'Vibration Control', 'Delay timer',
                   'Fixative Required', 'Handle Grips', 'Charger Included', 'Rise', 'Oil resistant',
                   'Integrates with Cell Phone', 'Installation kit included', 'Air compressor included',
                   'Short cycle protection', 'Mildew Resistant', 'Flashing light', 'Adjustable Fit',
                   'Compatible with Light Kits', 'Leather Palm', 'Colored', 'Lead Free', 'Adjustable Spray Pattern',
                   'Heated', 'Fan deck', 'Post-emergent', 'Power Take Off', 'Bypass valve',
                   'Batteries Included for Receiver', 'Padding Included', 'Ionic', 'Load Strap',
                   'Reinforced back pockets', 'Glue Down Allowed?', 'MP3 input', 'Covert', 'Waterproof camera head',
                   'Padded handle', 'Horizontal Tilt', 'Additional tips included', 'Motion-Activated Light', 'Convex',
                   'Aerosol', 'Casters', 'Offset Flange', 'Aerator', 'Contains Latex', 'Value Pack', 'Locking Lid',
                   'Aromatherapy reservoir', 'Braille Lettering', 'Mulching Capability', 'Custom Door Panel Ready',
                   'Commercial Application Requires Glue Down?', 'Built-In DVD Player', 'Textured',
                   'Cool-Touch Handles', 'Remote light control for home', 'Pulse Control', 'Transformer Included',
                   'Adjustable angle', 'Liner Included', 'Flexible Pitch', 'Ribbed', 'Filling panels included',
                   'Charge controller included', 'Fire retardant', 'Knob or pull included', 'Diverter Kit Included',
                   'Mixing nozzle included', 'Padded Grip', 'Odorless', 'Canopy', 'Replaceable LED Module',
                   'Vacuum breaker included', 'Included Accessories', 'Bendable head', 'Keypad', 'Louvered',
                   'High/low switch', 'Feet', 'Turntable', 'Waterproof Receiver', 'Double sided', 'Atomic Clock',
                   'Quick-Release Tension', '3-Way', 'Adapter(s) included', 'Headlights', 'Counter Depth (Yes/No)',
                   'Grounded', 'Adjustable straps', 'Floor drain required', 'Drain hose connection', 'Water repellent',
                   'Multiple base', 'File Storage', 'Remote control included', 'Flame Resistant', 'Low Odor',
                   'Extensions available', 'Accepts CDs', 'Fuel Gauge', 'Non-stick', 'Flammable liquid/fuel storage',
                   'Insect screen included', 'Breathable', 'Vertical-Horizontal Convertible', 'Light component',
                   'Blade Brake', 'Foldable handle', 'Lined', 'Peel-and-stick backing', 'Decorative',
                   'On/Off Switch for Laser Sight', 'Mixing nozzles included', 'Inflatable', 'Countertop',
                   'Molded-in brackets', 'Adjustable blade', 'Drawer', 'Tinted', 'Hood',
                   'Interchangeable nozzle connections', 'Meets MUTCD specs', 'Heat-resistant handles',
                   'Roller attachment', 'Resettable-customized combinations', 'Permanent', 'Temperature Display',
                   'Tilting cargo bed', 'Filter/bulb monitor', 'Plug-In', 'Moss control', 'Assembly Required',
                   'HUD Approved', 'Safe for garbage disposals', 'Fillet included', 'Water Resistant',
                   'Built-In Accessory Storage', 'California Title 20 Compliant', '2-Way Intercom Camera',
                   'Self drilling', 'Built-in chipper/mulcher', 'HDMI Out', 'Marine', 'Kit with accessories',
                   'Side sprayer', 'Folding Handle', 'Self closing', 'Broiler', 'Chair Back', 'Screen panels included',
                   'Armrests', 'Variable pin penetration', 'Bottom/End Suction', 'UL Listing', 'Collapsible',
                   'Outgoing mail indicator', 'Sound Lights', 'Pilot Light', 'Rubber tip', 'Pool Cover Pump',
                   'Infrared', 'Battery Charger Included', 'Learning', 'Able to be sharpened', 'Winding drum',
                   'Locking system', 'Submersible', 'Planter Included', 'Safe for Edibles', 'Pattern',
                   'Thermostatic Control', 'Knuckle strap/protection', 'Flashing', 'Buzzer Only', 'Fail-safe operation',
                   'Adjustable cutting height', 'Optical Zoom', 'Monitor Night Light', 'Shoulder strap',
                   'Wall Plate Included', 'Removes viruses', 'Modular', 'Reversible', 'Nonstick surface', 'Swivel head',
                   'Permanent settings memory', 'Rust dissolving additives', 'Alarm', 'Audio alert',
                   'Chain tension adjustment', 'Footing Required', 'Heat required', 'Clutch', 'Flange Included',
                   'Timer', 'Attached suspenders', 'Porch Included', 'Browning Control', 'Concea led hinge',
                   'Grounded Receptacle', 'HDMI In', 'Wrinkle resistant', 'Insulated Handle', 'Power clean',
                   'Wireless Speakers', 'Semiconductor Gas Sensor', 'Adjustable closing speed', 'Expandable',
                   'Toggle Switch', 'Built-in firewall', 'Hand Shower Included', 'Storm guard',
                   'Paint sprayer compatible', 'Hardware Required', 'Hose guide mounting brackets', 'D-Rings Included',
                   'Removable Blade', 'Ceramic Heating Element', 'Low-E (Y,N)', 'Fertilizer component',
                   'Reinforced palm/palm pad', 'Painted', 'Recommended For Powered Fish Tapes', 'High mileage',
                   'Rod Included', 'Cordless', 'Automatic Needle Threading', 'Chain Included', 'Integral lead storage',
                   'Paint & Primer in One', 'Solvent resistant', 'Made in USA', 'Polarized', 'Caulkless',
                   'Variable speeds', 'Caps Included', 'Installed privacy slats', 'Safe on any surface', 'Collegiate',
                   'Swivel base', 'Dimmable', 'Corrugated', 'Rear Panel', 'Low Oil Shut Down',
                   'Filter change indicator', 'Self-Cleaning', 'Slide-Out Crumb Tray', 'Electronic',
                   'Magnetic disc brakes', 'Removable Pan(s)', 'HE Formulated', 'Built-In Media Storage',
                   'Rain Switch Included', 'Oil Free Compressor', 'Spandex/stretchable back', 'Programmable',
                   'Ice/Water Dispenser', 'Brand compatibility', 'Escutcheon Included', 'Handle',
                   'Compatible with a knee kicker', 'Washers Included', 'Soap Dispenser Included', 'Pressure balance',
                   'Gripper back', 'Electrical brake', 'Background', 'Tooth extension included', 'Applicator in lid',
                   'Crib Mountable', 'Organic', 'Reverse', 'Latches', 'Shelf Included', 'Security Device Use',
                   'Cleaner Impregnated', 'Cord Slot/ Runner', 'Clear-coat safe', 'Scald guard', 'Bagless', 'Curling',
                   'Removes particulates', 'Shatter Resistant', 'Hooks', 'Battery Type Required', 'Pole included',
                   'Stacking Kit Model#', 'Smoker chip tray included', 'Open/close function', 'RMS',
                   'Faucet adapter included', 'Two Part', 'Powered', 'Seat Included', 'Hidden Controls',
                   'Battery backup', 'Vapor Retardent', 'Adjustable jet flow', 'Peel backing', 'Pump Included',
                   'Installation Required', 'Tray Included', 'Self Cleaning', 'Adjustable Lamp Head',
                   'Articulating head', 'Rain gauge', 'Cap/top included', 'Twin wheel', 'Pre-Filled',
                   'Bag Change/Receptacle Full Indicator', 'Twin Stack Tank Configuration', 'Replaceable cutting blade',
                   'Self-Feeding', 'Built-in Answering Machine', 'Power Surge Protection', 'Overspeed protection',
                   'Sheath Included', 'Grill', 'Solar powered', 'Full-Size Feed Tube', 'Key lock', 'Includes felt pad',
                   'Keyed', 'Floats in water', 'Wheels', 'Fixed', 'Adjustable Flame', 'Ratchet included',
                   'Cycle - Allergen', 'No-tool adjustment', 'Thermostat', 'Drift cutters', 'Storage pocket',
                   'Camera Use', 'Provides Extension', 'Hose Port', 'Built-in carbon filter', 'Collar included',
                   'Rear access', 'Magnifying', 'Pneumatic Tires', 'Cool Air Button', 'Side hole included',
                   'Extension(s) included', 'Hanging strap', 'Rust Resistant', 'Impact Resistant', 'Non-Clogging',
                   'Container included', 'Bolts to Floor', 'Sanded', 'Reflective', 'Vehicular Traffic Rated',
                   'Mounting Bracket Included', 'Bifocal', 'Swinging Gate', 'Handle grip', 'Newspaper receptacle',
                   'Brad Nailer Included', 'Flocked/frosted', 'Blade guard system', 'Reverse Airflow',
                   'Replaceable Battery', 'Lockable Door/Gate Latch', 'Detergent tank', 'Digests Paper', 'Light',
                   'Light included', 'High Efficiency Washer', 'Temperature limit adjustment',
                   'Securable/adjustable wrist cuff', 'Multifunctional spatula', 'Strippable', 'Cleanout',
                   'Tamper Resistant', 'Primer required', 'Duct System', 'Candle included', 'Winter blade',
                   'Drain Pipe Required', 'Removable guide', 'Internal Control', 'Reversible Two-Sided Blades',
                   'Paint/stain stripper', 'Rolling code technology', 'Convection', 'Storage Bag Included',
                   'Pivot included', 'Leader Hose Included', 'Audio Alert/Siren', 'Sealer', 'Tread',
                   'Cool-Touch Exterior', 'Adjustable Valve', '16 Gauge Finish Nailer Included', 'Shatterproof',
                   'Serrated', 'Degreaser', 'Hand-Set Speaker Phone', 'Shock Absorbent', 'Hutch Included',
                   'Chimnea/ Pole Mount', 'Automatic volume limiter', 'Track included', 'Battery level indicator',
                   'Air vents', 'Predator Guard', 'Gold Plated', 'Thermometer Included', 'Point-Of-Use', 'Solar',
                   'Replaceable pads', 'Lid Strike', 'Drain connector included', 'Adjustable nozzle', 'Quick release',
                   'Welded Seams', '3-Axis Adjustable Camera', 'Fasteners included', 'Hanging Rails Included',
                   'Protects pipes from corrosion', 'Magnetic', 'Hook(s)', 'Roller included', 'Waterless',
                   'Built-In Shower Seat', 'Dishwasher drain connection', 'Adjustable flow system',
                   'Attic ventilation required', 'Windshield', 'Thermal Overload Protection',
                   'High Temperature Resistance', 'Removable Shelves', 'Grill surface', 'Built-In Radio', 'Engravable',
                   'Formaldehyde Free', 'Removable Cooking Surface', 'Tintable', 'Dual Fan Heads',
                   'Carpet Height Adjustments', 'Paintable', 'Ratcheting', 'Detachable Spindle', 'Solids Handling',
                   'Fire Block Rated', 'Mulch and Soil Council Certified', 'Lighting Included',
                   'Pull-Out Wiriting Board', 'Saucer Included', 'Ceramic disk valves', 'Heat Resistant Handles',
                   'Predrilled holes', 'Under the Counter Installation', 'Work light included', 'Double frame',
                   'Animated', 'Calendar', 'Odd/Even', 'Forward/Reverse Rocker Switch', 'Rekeyable',
                   'Overload limiting clutch', 'Run-Through Protection', 'Downdraft Exhaust', 'Check valve', 'Radio',
                   'Hospital Grade', 'Hardware Included', 'Washable/Reusable', 'Quartz movement', 'Storm window',
                   'Pivot Rod', 'Tips Included', 'Floor Options', 'Parquet', 'Conduit Reaming', 'Attachable/detachable',
                   'Covered', 'Spindle Lock', 'Tumbler Included', 'Lighting Kit', 'Ethernet Connection',
                   'Pressure Relief Valve', 'Self-adhesive', 'Pro Pack', 'Precut lines', 'Euro Top', 'Removes bacteria',
                   'Lockable', 'Pressure Sensitive Pen', 'Swivel load hook', 'Screws Included', 'Padded Shoulders',
                   'Wireless keypad', 'Tests Resistance', 'Upholstered Cover', 'Doors Included', 'Wireless Cameras',
                   'Snap-On Installation', 'Motion Sensor', 'Rotates for Cleaning', 'Child safe', 'Lift-out tray',
                   'Water Soluble', 'Wrap cover/lens included', 'Swivel cap', 'Dries clear', 'Adjustable heads',
                   'Washer Interior Light', 'Steam Vents', 'Displays humidity', 'Adjustable stream',
                   'Graduation markings', 'Multiple sizes', 'Printed on Both Sides', 'Laser level included',
                   'Elevation Sensor', 'Handheld', 'Displays wind chill', 'Prestarted nails', 'Rot Resistant',
                   'Manual Option', 'PC / DVI Input', 'Hammer included', 'Double Thick Seal', 'Wired edge',
                   'Slip-resistant feet', 'Post included', 'Pocket', 'Unit can be Installed Over a Wall Oven',
                   'Separatable rings', 'Rotating Head', 'Bagger', 'Backlit', 'Drought Tolerant', 'Pillowtop',
                   'Wide Angle', 'Edge Cleaner', 'Digital Bell', 'Forecast ability', 'Removable Drawer',
                   'Telescoping Handles', 'Automatic defrost control', 'Extra Pins/Probes Included',
                   'Detachable Water Tank', 'Carrying handle', 'Multiple size sheet options', 'Cooling fan',
                   'Prevention formula', 'Intercom', 'Works with all water softeners', 'Slip Resistant Surface',
                   'Work Light', 'Slim Profile', 'Portable', 'Compressor required', 'Adjustable Shelves',
                   'Severe weather ready', 'Recessed', 'Adjustable Fork Width', 'Free Pick-up/Delivery Service', 'Bake',
                   'Vented', 'Synthetic', 'Removes dust', 'Built-in stand', 'Motor Included', 'Eye shields',
                   'Built-In backrest', 'Installation Hardware Included', 'Cushioning', 'Limited edition', 'Jacketed',
                   'Warning Tone', 'Tests continuity', 'Coated', 'Removable cover', 'Dual Flush',
                   'Low voltage audible alarm', 'Commercial', 'Powered cable feed', 'Self-Priming', 'UVC lamp',
                   'Padding Required?', 'Reclining', 'GFIC plug', 'Nonslip grip', 'Hands-Free', 'Linkable',
                   'Ready To Use', 'Clockwise rotation', 'Antivibration', 'Required door', 'Side Door',
                   'Shock-Absorbing Bumper', 'Additional hose attachments included', 'Multi-pack keyed alike',
                   'Includes epoxy cartridge(s)', 'Fittings Included', 'Solar Powered', 'Adjustable Height',
                   'Fuse included', 'Removable Drip/Crumb Tray', 'Lockable Flap/Panel', 'Universal',
                   'Phone/Data Protection', 'Weighted Base', 'Removable Filter', 'Attachable', 'Double Rod',
                   'Drain Plug', 'Foldable', 'Coated Blades', 'Two-sided', 'Weather-resistant case',
                   'Adjustable mounting positions', 'Locking', 'Rinse required', 'Antimicrobial', 'Programmable Timer',
                   'Safe for Pets', 'Noncontact detection', 'Swivel', 'Universal Remote', 'Can be wallpapered',
                   'Rolling', 'Random orbit', 'Adjustable ball mount', 'Sandbox Safe', 'Delay Start Option', 'Reusable',
                   'Non-Stick Interior', 'Safety sensor', 'Prevents/blocks mold and mildew', 'Shielded',
                   'Includes Flapper', 'Belt clip', 'Plug and play', 'Flickering', 'Handle Included', 'USB Connection',
                   'Floating Hinge', 'Continuous on technology', 'Lateral Filing', 'Adjustable Speed', 'Audible Alert',
                   'Amplifier', 'Kink Resistant', 'Smart Timer', 'Side Sprayer Included', 'Removes mold',
                   'Vertical Tilt', 'Can Be Cut To Length', 'String trimmer conversion capable', 'Convertible to push',
                   'Digital Control', 'Mold and mildew resistant', 'Audio Recording', 'Sprays on clear',
                   'Termite Resistant', 'High-Efficiency Detergent Required', 'Paintable/Stainable', 'Bleach Added',
                   'Detachable Hose', 'Flow Restrictor', 'Solder included', 'Audio input connector', 'Tempered Blade',
                   'Sprayer head included', 'Drainage Holes', 'Bypass Valve Included', 'Contains nuts', 'Safety cuff',
                   'Double loops', 'Convertible to post mount', 'End of Cycle Alarm', 'Vented nozzle', 'Spatter shield',
                   'Beveled frame', 'Welded', 'Antiscratch lens coating', 'Detachable Base', 'Hand brakes',
                   'Touch Screen', 'Media Fireplace', 'Input jack', 'Depth Measurement System', 'Adjustable',
                   'Pre-Drilled Holes', 'Retractable Cord', 'Adjustable Front Straps', 'Dishwasher Safe',
                   'Washer - Delay Start', 'Scraper', 'Perforated', 'Reception Amplified', 'Filter unit included',
                   'Flux required', 'Switch On Power Unit', 'Coated tines', 'Steam Function', 'Unions Included',
                   'Removable lid', 'Swivel front wheel', 'Air flow control system', 'Side Burner', 'Adjustable Racks',
                   'Side Handle Included', 'Accepts Paper Clips', 'Bypass Valve Required', 'Serrated edges',
                   'Beveled Edges', 'Ventilation holes', 'Can Be Sharpened', 'Depth markings', 'Lock Indicator',
                   'Low battery shutdown', 'Displays barometric pressure', 'Indicator light(s)', 'Steam Control',
                   'USB Port', 'Class A Fire Rating', 'Station included', 'Tank included', 'Kabob Skewers',
                   'Expandable Power Outlet', 'Adjustable roof vent', 'Built-In Drain', 'Vertical/overhead use',
                   'Underlayment Required', 'Finger attachment', 'Pool approved', 'Rated for in-wall', 'Hot Water Use',
                   'Vandal resistant', 'Self-rinsing', 'Nightlight', 'Trend lines', 'Flow Control Valve',
                   'Digital Display', 'Pressure Treated', 'Battery included', 'Keep Warm Setting', 'Weight tray',
                   'Sensor Cook', 'Handles included', 'Electric Brake', 'Heat resistant',
                   'Automatic line-advance system', 'Roll Top Enclosure', 'Disposable', 'Blade storage in handle',
                   'Spray application', 'Window(s)', 'Flat-Free Tires', 'Swivel Seat', 'Steam head included',
                   'Shrouded shackle', 'Grinding Wheel Included', 'Water repellant', 'Manual Release Handle',
                   'Ring storage', 'Glow-In-The-Dark', 'Illuminated buttons/display', 'Timer Included', 'Kink free',
                   'No Tool Blade Change', 'Preprogrammed Embroidery Designs', 'Programmable thermostat',
                   'Pre-Assembled', 'Adhesive backing', 'Compatible with tankless air compressor', 'Non-Glare',
                   'Foot pedal on/off', 'Springs Included', 'Base Included', 'Closure caps included',
                   'Sunlight / UV Resistant', 'Moisture control included', 'Biodegradable', 'Built-in cutter',
                   'Integrated electrical outlet', 'Ambient Concentration Reset', 'Internal Filtered Water Dispenser',
                   'Carpet Attachment', 'Inverter included', 'Catch included', 'Microwave safe', 'Pitch Adjustment',
                   'Clips/rings included', 'Adjustable flow rate', 'Phonograph (Turntable) Input', 'High wheel',
                   'Includes fuse tester/puller', 'With Tank', 'Cut for Custom Fit', 'Waterfall Pump',
                   'Pilot Bit Included', 'Letters/numbers included', 'Adjustable Temperature Control', 'UV Protection',
                   'Pump Hookup Included', 'Accessory outlets', 'Wall panels included', 'Loveseat Included',
                   'Built-in drip pan', 'Lockable Blade', 'Weatherable', 'Controls broad leaf', 'Reversible Door(s)',
                   'Pre-decorated', 'Patch cords included', 'Telescoping control rod', 'Cleated Bottom', 'Oscillating',
                   'Griddle', 'Built-In Storage', 'Optional mirrored light fixture', 'Antivibration handle',
                   'Dielectric', 'Retractable Blade', 'Hearing Aid Use', 'Media Center', 'Full Sized Keg',
                   'Built-in Caller ID', 'Fuel Included', 'Weather-resistant hardware', '2-piece drawbar',
                   'Pond Included', 'Nonabrasive', 'Hurricane rated', 'Adjustable Volume', 'Reinforced knee',
                   'Use on vegetables', 'Commercial Grade', '18 Gauge Finish Stapler Included', 'Faucet Included',
                   'With screw', 'Mechanical Bell', 'Ratcheted', 'Built-in backsplash', 'In-Ground Installation',
                   'Washable', 'Fertilizer Enriched', 'Dual tine tips', 'High Definition Compatible',
                   'Battery Test Function', 'Adjustable jet direction', 'Adjustable spray tip',
                   'Automatic depth control', 'Safe Around Vegetation', 'Automatic Temperature Compensation', 'Pump',
                   'Side Handle', 'Replacement washers included', 'Enclosed Holder', 'Drill Attachable', 'Twinkling',
                   'UV Stabilized', 'Drill bit included', 'Moisture Resistant', 'Deck cleanout', 'Remote controlled',
                   'Onboard Battery Charger', 'Floor vase', 'Fan', 'Automatic Cover System', 'Prefinished',
                   'Skin Care Tool Included', 'Adjustable grate height', 'Speed Dial', 'Concentrated', 'LED Display',
                   'Moisture Control', 'Vinyl Siding Institute rated', 'Coil Cord', 'Retraction Spring',
                   'Adjustable session settings', 'Tall Tub', 'Conversion kit', 'Mountable inside steam shower/room',
                   'Interchangeable leads', 'Electrical wire included', 'Built-in lighting', 'Lights',
                   'Adjustable Detection Sensitivity', 'Reinforced front panels', 'Wall mounted',
                   'Mounting Hardware Included', 'Tone-Only Mode', 'Kits', 'Threaded with Wiring',
                   'Safety chain included', 'Driver bit included', 'Arbor included', 'Carabiner(s)',
                   'Moldable nose piece', 'Stainable/Paintable', 'Automatic Shut-off', 'Adjustable screen length',
                   'Fan-only option', 'Liner Locking Feature', 'Friction Fit', 'Ice auger conversion capable',
                   'Key chuck included', 'Drain Included', 'Heat Thermometer', 'Tank Pressure Gauge', 'Main Breaker',
                   'Antitippers', 'Power On Light Indicator', 'Crush Resistant', 'Adjustable Mirror',
                   'Adjustable Length', 'Sound absorbing', 'Fasteners and gasket included', 'Glazed',
                   'Anti-freeze setting', 'Built-In Timer', 'Two-way reflector', 'Mallet Required',
                   'On Indicator Light', 'Cushioned', 'Digital Temperature Control', 'Toothbrush Holder', 'Swivel Lock',
                   'Base Required', 'Squirrel Guard', 'Air Volume Control', 'Pause/Interrupt',
                   'Cleaning Chemicals Included', 'Automatic Calibration', 'Tounge and Groove', 'Run-Dry Capable',
                   'Preconfigured', 'Adjustable Arm Rests', 'Stain Resistant', 'Bullet Proof', 'Flashing Included',
                   'Onboard storage', 'Skid Resistant', 'Noise insulation', 'Heater included', 'Mounting screw hole',
                   'Volume control', 'Pre-Lit', 'Automatic shutoff', 'Scrubbable', 'Microcontroller Circuit',
                   'Control Box Included', 'Socket(s) included', 'Removable', 'Propane (LP) Conversion Kit Included',
                   'Programmable Correction', 'Entry Rollers', 'Diverter on valve', 'Pad lockable', 'Tumbler Holder',
                   'Environmentally Safe', 'High-Contrast', 'Underground rated', 'Nondrip nozzle',
                   'Adjustable limit switches', 'Suitable for Metal Only', 'Variable Speed', 'Clock', 'Secure open lid',
                   'Lighter Fluid Required', 'Reverse Polarity Protection', 'Safety guards', 'Striped',
                   'Non-slip hand grip', 'Wheeled', 'Packing Material Included', 'Temperature Control', 'Comfort',
                   'Scrubs Tile', 'Adjustable Lumbar Support', 'Dishwasher Safe Parts', 'Box Included', 'Multi-family',
                   'Upholstered Seat', 'Complete Toilet Kit', 'Saw Attachment', 'Dry erase marker included',
                   'Removes mold and mildew', 'Deck Use', 'Fitted', 'Surface Mount', 'Humidity Control',
                   'Safeguard Guides', 'Energy Saving', 'Supply lines included', 'Corner roller',
                   'Built-in Water Filter', 'Case Included', 'Fully adjustable', 'Pre-Charged',
                   'Turntable On/Off Option', 'Twist and lock design', 'Holder Included', 'USB', 'Mold Resistant',
                   'Mirrored', 'Dispenser Included', 'Adjustable wheel', 'Cord Included',
                   'U.S. Postmaster General approved', 'Tamper end', 'GFIC', 'Kid and Teen Theme', 'Controls crabgrass',
                   'Flame retardant', 'Extendable Table', 'Stainable/paintable handrail', 'Insulated',
                   'Tool-free adjustments', 'Pre-diluted', 'Movable', 'Dust Compression System', 'Wireless',
                   '15 Gauge Finish Nailer Included', 'Mold/mildew resistant', 'Light transmittal',
                   'Adjustable Backrest', 'Yaw Adjustment', 'Includes wireless remote control', 'UV/sunlight resistant',
                   'Water based', 'Lever Compatible', 'Adjustable brightness', 'Tapered', 'Care Instructions',
                   'Remote Control', 'Musical', 'Zippered Top', 'Slip-resistant sole', 'Interior Light',
                   'ENERGY STAR Certified', 'Door Bell Included', 'Forged Blade', 'Thermal', 'Structural',
                   'Catch pan included'}


def get_syn(w):
    return additional_syn.get(w, set([]))
