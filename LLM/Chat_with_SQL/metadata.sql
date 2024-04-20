CREATE TABLE actor (
	actor_id SERIAL NOT NULL,
	first_name VARCHAR(45) NOT NULL,
	last_name VARCHAR(45) NOT NULL,
	last_update TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
	CONSTRAINT actor_pkey PRIMARY KEY (actor_id)
)
CREATE TABLE address (
	address_id SERIAL NOT NULL,
	address VARCHAR(50) NOT NULL,
	address2 VARCHAR(50),
	district VARCHAR(20) NOT NULL,
	city_id SMALLINT NOT NULL,
	postal_code VARCHAR(10),
	phone VARCHAR(20) NOT NULL,
	last_update TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
	CONSTRAINT address_pkey PRIMARY KEY (address_id),
	CONSTRAINT fk_address_city FOREIGN KEY(city_id) REFERENCES city (city_id)
)
CREATE TABLE category (
	category_id SERIAL NOT NULL,
	name VARCHAR(25) NOT NULL,
	last_update TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
	CONSTRAINT category_pkey PRIMARY KEY (category_id)
)
CREATE TABLE city (
	city_id SERIAL NOT NULL,
	city VARCHAR(50) NOT NULL,
	country_id SMALLINT NOT NULL,
	last_update TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
	CONSTRAINT city_pkey PRIMARY KEY (city_id),
	CONSTRAINT fk_city FOREIGN KEY(country_id) REFERENCES country (country_id)
)
CREATE TABLE country (
	country_id SERIAL NOT NULL,
	country VARCHAR(50) NOT NULL,
	last_update TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
	CONSTRAINT country_pkey PRIMARY KEY (country_id)
)
CREATE TABLE customer (
	customer_id SERIAL NOT NULL,
	store_id SMALLINT NOT NULL,
	first_name VARCHAR(45) NOT NULL,
	last_name VARCHAR(45) NOT NULL,
	email VARCHAR(50),
	address_id SMALLINT NOT NULL,
	activebool BOOLEAN DEFAULT true NOT NULL,
	create_date DATE DEFAULT ('now'::text)::date NOT NULL,
	last_update TIMESTAMP WITHOUT TIME ZONE DEFAULT now(),
	active INTEGER,
	CONSTRAINT customer_pkey PRIMARY KEY (customer_id),
	CONSTRAINT customer_address_id_fkey FOREIGN KEY(address_id) REFERENCES address (address_id) ON DELETE RESTRICT ON UPDATE CASCADE
)
CREATE TABLE film (
	film_id SERIAL NOT NULL,
	title VARCHAR(255) NOT NULL,
	description TEXT,
	release_year INTEGER,
	language_id SMALLINT NOT NULL,
	rental_duration SMALLINT DEFAULT 3 NOT NULL,
	rental_rate NUMERIC(4, 2) DEFAULT 4.99 NOT NULL,
	length SMALLINT,
	replacement_cost NUMERIC(5, 2) DEFAULT 19.99 NOT NULL,
	rating mpaa_rating DEFAULT 'G'::mpaa_rating,
	last_update TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
	special_features TEXT[],
	fulltext TSVECTOR NOT NULL,
	CONSTRAINT film_pkey PRIMARY KEY (film_id),
	CONSTRAINT film_language_id_fkey FOREIGN KEY(language_id) REFERENCES language (language_id) ON DELETE RESTRICT ON UPDATE CASCADE
)
CREATE TABLE film_actor (
	actor_id SMALLINT NOT NULL,
	film_id SMALLINT NOT NULL,
	last_update TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
	CONSTRAINT film_actor_pkey PRIMARY KEY (actor_id, film_id),
	CONSTRAINT film_actor_actor_id_fkey FOREIGN KEY(actor_id) REFERENCES actor (actor_id) ON DELETE RESTRICT ON UPDATE CASCADE,
	CONSTRAINT film_actor_film_id_fkey FOREIGN KEY(film_id) REFERENCES film (film_id) ON DELETE RESTRICT ON UPDATE CASCADE
)
CREATE TABLE film_category (
	film_id SMALLINT NOT NULL,
	category_id SMALLINT NOT NULL,
	last_update TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
	CONSTRAINT film_category_pkey PRIMARY KEY (film_id, category_id),
	CONSTRAINT film_category_category_id_fkey FOREIGN KEY(category_id) REFERENCES category (category_id) ON DELETE RESTRICT ON UPDATE CASCADE,
	CONSTRAINT film_category_film_id_fkey FOREIGN KEY(film_id) REFERENCES film (film_id) ON DELETE RESTRICT ON UPDATE CASCADE
)
CREATE TABLE inventory (
	inventory_id SERIAL NOT NULL,
	film_id SMALLINT NOT NULL,
	store_id SMALLINT NOT NULL,
	last_update TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
	CONSTRAINT inventory_pkey PRIMARY KEY (inventory_id),
	CONSTRAINT inventory_film_id_fkey FOREIGN KEY(film_id) REFERENCES film (film_id) ON DELETE RESTRICT ON UPDATE CASCADE
)
CREATE TABLE language (
	language_id SERIAL NOT NULL,
	name CHAR(20) NOT NULL,
	last_update TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
	CONSTRAINT language_pkey PRIMARY KEY (language_id)
)
CREATE TABLE payment (
	payment_id SERIAL NOT NULL,
	customer_id SMALLINT NOT NULL,
	staff_id SMALLINT NOT NULL,
	rental_id INTEGER NOT NULL,
	amount NUMERIC(5, 2) NOT NULL,
	payment_date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
	CONSTRAINT payment_pkey PRIMARY KEY (payment_id),
	CONSTRAINT payment_customer_id_fkey FOREIGN KEY(customer_id) REFERENCES customer (customer_id) ON DELETE RESTRICT ON UPDATE CASCADE,
	CONSTRAINT payment_rental_id_fkey FOREIGN KEY(rental_id) REFERENCES rental (rental_id) ON DELETE SET NULL ON UPDATE CASCADE,
	CONSTRAINT payment_staff_id_fkey FOREIGN KEY(staff_id) REFERENCES staff (staff_id) ON DELETE RESTRICT ON UPDATE CASCADE
)
CREATE TABLE rental (
	rental_id SERIAL NOT NULL,
	rental_date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
	inventory_id INTEGER NOT NULL,
	customer_id SMALLINT NOT NULL,
	return_date TIMESTAMP WITHOUT TIME ZONE,
	staff_id SMALLINT NOT NULL,
	last_update TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
	CONSTRAINT rental_pkey PRIMARY KEY (rental_id),
	CONSTRAINT rental_customer_id_fkey FOREIGN KEY(customer_id) REFERENCES customer (customer_id) ON DELETE RESTRICT ON UPDATE CASCADE,
	CONSTRAINT rental_inventory_id_fkey FOREIGN KEY(inventory_id) REFERENCES inventory (inventory_id) ON DELETE RESTRICT ON UPDATE CASCADE,
	CONSTRAINT rental_staff_id_key FOREIGN KEY(staff_id) REFERENCES staff (staff_id)
)
CREATE TABLE staff (
	staff_id SERIAL NOT NULL,
	first_name VARCHAR(45) NOT NULL,
	last_name VARCHAR(45) NOT NULL,
	address_id SMALLINT NOT NULL,
	email VARCHAR(50),
	store_id SMALLINT NOT NULL,
	active BOOLEAN DEFAULT true NOT NULL,
	username VARCHAR(16) NOT NULL,
	password VARCHAR(40),
	last_update TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
	picture BYTEA,
	CONSTRAINT staff_pkey PRIMARY KEY (staff_id),
	CONSTRAINT staff_address_id_fkey FOREIGN KEY(address_id) REFERENCES address (address_id) ON DELETE RESTRICT ON UPDATE CASCADE
)
CREATE TABLE store (
	store_id SERIAL NOT NULL,
	manager_staff_id SMALLINT NOT NULL,
	address_id SMALLINT NOT NULL,
	last_update TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
	CONSTRAINT store_pkey PRIMARY KEY (store_id),
	CONSTRAINT store_address_id_fkey FOREIGN KEY(address_id) REFERENCES address (address_id) ON DELETE RESTRICT ON UPDATE CASCADE,
	CONSTRAINT store_manager_staff_id_fkey FOREIGN KEY(manager_staff_id) REFERENCES staff (staff_id) ON DELETE RESTRICT ON UPDATE CASCADE
)

-- address.city_id can be joined with city.city_id
-- city.country_id can be joined with country.country_id
-- customer.address_id can be joined with address.address_id
-- film.language_id can be joined with language.language_id
-- film_actor.actor_id can be joined with actor.actor_id
-- film_actor.film_id can be joined with film.film_id
-- film_category.category_id can be joined with category.category_id
-- film_category.film_id can be joined with film.film_id
-- inventory.film_id can be joined with film.film_id
-- inventory.store_id can be joined with store.store_id
-- payment.customer_id can be joined with customer.customer_id
-- payment.rental_id can be joined with rental.rental_id
-- payment.staff_id can be joined with staff.staff_id
-- rental.customer_id can be joined with customer.customer_id
-- rental.inventory_id can be joined with inventory.inventory_id
-- rental.staff_id can be joined with staff.staff_id
-- staff.address_id can be joined with address.address_id
-- store.address_id can be joined with address.address_id
-- store.manager_staff_id can be joined with staff.staff_id

