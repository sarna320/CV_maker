import requests
from langchain_community.document_loaders import UnstructuredHTMLLoader
from typing import Optional, List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.llms.ollama_functions import OllamaFunctions
import os
import json
from langchain_openai import ChatOpenAI
from tool import create_example_messages
from utility import extract_secret


class JobOffer(BaseModel):
    """Information about a job offer."""

    position_name: Optional[str] = Field(
        default=None, description="The name of the position."
    )
    company_name: Optional[str] = Field(
        default=None, description="The name of the company offering the job."
    )
    position_level: Optional[str] = Field(
        default=None, description="The level or seniority of the position."
    )
    specializations: Optional[List[str]] = Field(
        default=None, description="The specializations required for the position."
    )
    skills: Optional[List[str]] = Field(
        default=None, description="The skills required for the position."
    )
    about_position: Optional[str] = Field(
        default=None, description="A brief description about the position."
    )
    responsibilities: Optional[List[str]] = Field(
        default=None, description="The responsibilities associated with the position."
    )
    requirements: Optional[List[str]] = Field(
        default=None, description="The requirements needed to qualify for the position."
    )
    about_company: Optional[str] = Field(
        default=None,
        description="A brief description about the company offering the job.",
    )


examples = [
    (
        "Przejdź do treści ogłoszenia  Przejdź do panelu aplikowania  Przejdź do panelu bocznego  Przejdź do stopki  Niestety, nie wspieramy Twojej przeglądarki  Niestety nie wpieramy Twojej przeglądarki co może znacznie wpłynąć na poprawne ładowanie skryptów strony.  nowość  Oferty pracy  Pобота  Profile pracodawców  Moja kariera  Moje konto  Dla firmDodaj ogłoszenie  Oferta pracy  Start  Szukaj  Strefa ofert  Konto  Menu  AI Junior Engineer - Internship  BAT DBS Poland Sp. z o.o.About the company  WarszawaWarszawa, Masovian  valid for 19 daysto 07 July 2024  internship / apprenticeship contract  trainee  full office work  remote recruitment  Specializations:AI/ML  Technologies we use  Expected  Python  C++  C#  Java  About the project  We are thrilled to launch the AI Talents – internship program, designed for students, eager to dive into the world of Artificial Intelligence.  This unique opportunity allows you to develop cutting-edge AI skills while immersing yourself in our vibrant people, culture, and technology environment.  Join us to collaborate with industry experts, work on real-world projects, and be part of a community that fosters innovation and growth.  Don’t miss your chance to shape the future with AI—apply now and unleash your potential!    Positive Attitude: We're seeking individuals with a proactive mindset, a collaborative spirit, and a passion for learning and growth.    Tech-Savvy Enthusiast: Knowledge of computer skills and some experience with programming are preferred, along with a genuine eagerness to explore and work with AI, driven by curiosity and a desire to innovate.    Analytical Skills: Ideal candidates have a solid foundation in mathematics and problem-solving, essential for mastering AI technologies.  Your responsibilities  Data Collection and Preparation: Assist in gathering, cleaning, and organizing datasets required for AI model training and evaluation.  Model Development: Support senior engineers in developing and fine-tuning AI models using appropriate algorithms and techniques.  Testing and Validation: Conduct testing and validation of AI models to ensure accuracy, reliability, and performance.  Documentation: Maintain detailed documentation of AI models, processes, and findings to ensure clarity and reproducibility.  Code Implementation: Write efficient, maintainable code for AI applications, following best practices and coding standards.  Collaboration: Work closely with cross-functional teams, including data scientists, software engineers, and domain experts, to integrate AI solutions.  Research: Stay updated with the latest AI trends, tools, and technologies, and contribute to research activities as needed.  Troubleshooting: Identify and resolve issues related to AI model performance, data quality, and deployment challenges.  Our requirements  Basic familiarity with at least one programming language such as Python, C++/C#, or Java.  Some exposure to AI or machine learning projects through internships, academic projects, or personal projects.  Analytical skills and a basic understanding of mathematical concepts.  Knowledge of data structures and algorithms.  Ability to work with databases and perform basic data manipulation tasks.  Introductory coursework or certifications in AI, machine learning, or data science are a plus.  This is how we organize our work  This is how we work  agile  scrum  kanban  What we offer  Hands-On Experience with Cutting-Edge Technology: Gain practical experience working on AI projects using leading platforms such as Azure, enhancing your technical skills and knowledge.  Comprehensive Training and Development: Participate in ongoing training programs, workshops, and mentorship opportunities to deepen your understanding of AI concepts and tools.  Collaborative and Innovative Environment: Work alongside experienced professionals in a supportive, collaborative setting that fosters innovation, creativity, and career growth.  Role Positioning and Objectives  Reports to: Platform AI Manager    Reporting Level: Individual Contributor    Geographic Scope: local    Travel Required: no required  BAT DBS Poland Sp. z o.o.  Scroll to company's profile  WE ARE BAT DBS    At BAT DBS we are committed to our Purpose of creating A Better Tomorrow. This is what drives our people and our passion for innovation. See what is possible for you at BAT DBS Hub    - Global Employer with 220 DBS people inhouse  - Brands sold in over 200 markets, made in 44 factories in 42 countries  - Newly established Tech Hub building world-class capabilities for innovation in 4 strategic locations  - Diversity leader in the Financial Times and International Women’s Day Best Practice winner    BELONGING, ACHIEVING, TOGETHER    Collaboration, diversity, and teamwork underpin everything we do here at BAT. We know that collaborating with colleagues from different backgrounds is what makes us stronger and best prepared to meet our business goals. Come bring your difference!  Aplikuj  Aplikuj  Zapisz  Drukuj  Udostępnij  Praca  Warszawa  IT - Administracja  Administrowanie bazami danych i storage  Aplikuj  Zapisz  Drukuj  Udostępnij  AI Junior Engineer - Internship, Warszawa  Dla kandydatów  Pomoc  Pracuj w Grupie Pracuj  Festiwal Pracy JOBICON  Kalkulator godzinowy  Dla firm  Dodaj ogłoszenie  Konto pracodawcy  Pomoc dla firm  Porady dla firm  Porady rekrutacyjne  Grupa Pracuj  O nas  Centrum Prasowe  Reklama  Partnerzy  Archiwum ofert  Narzędzia  Kreator CV  Porady zawodowe  Kalkulator zarobków  Zarobki  The:blog  Nasze produkty  Pobierz aplikację  © Grupa Pracuj S.A.  Regulamin  Polityka Prywatności  Polityka plików cookies  Akt o usługach cyfrowych",
        JobOffer(
            position_name="AI Junior Engineer",
            company_name="BAT DBS Poland Sp. z o.o.",
            position_level="internship / apprenticeship contract",
            specializations=[
                "AI",
                "ML",
            ],
            skills=[
                "Python",
                "C++",
                "C#",
                "Java",
            ],
            requirements=[
                "Basic familiarity with at least one programming language such as Python, C++/C#, or Java.",
                "Some exposure to AI or machine learning projects through internships, academic projects, or personal projects.",
                "Analytical skills and a basic understanding of mathematical concepts.",
                "Knowledge of data structures and algorithms.",
                "Ability to work with databases and perform basic data manipulation tasks.",
                "Introductory coursework or certifications in AI, machine learning, or data science are a plus.",
            ],
            responsibilities=[
                "Data Collection and Preparation. Assist in gathering, cleaning, and organizing datasets required for AI model training and evaluation.",
                "Model Development Support. senior engineers in developing and fine-tuning AI models using appropriate algorithms and techniques.",
                "Testing and Validation. Conduct testing and validation of AI models to ensure accuracy, reliability, and performance.",
                "Documentation Maintain. detailed documentation of AI models, processes, and findings to ensure clarity and reproducibility.",
                "Code Implementation. Write efficient, maintainable code for AI applications, following best practices and coding standards.",
                "Collaboration. Work closely with cross-functional teams, including data scientists, software engineers, and domain experts, to integrate AI solutions.",
                "Research. Stay updated with the latest AI trends, tools, and technologies, and contribute to research activities as needed.",
                "Troubleshooting. Identify and resolve issues related to AI model performance, data quality, and deployment challenges.",
            ],
            about_position="Announcing the AI Talents internship program for students passionate about Artificial Intelligence! Develop AI skills, work on real projects, and collaborate with industry experts. We're seeking proactive, tech-savvy individuals with programming experience and strong analytical skills. Apply now to shape the future with AI!",
        ),
    ),
    (
        "Przejdź do treści ogłoszenia  Przejdź do panelu aplikowania  Przejdź do panelu bocznego  Przejdź do stopki  Niestety, nie wspieramy Twojej przeglądarki  Niestety nie wpieramy Twojej przeglądarki co może znacznie wpłynąć na poprawne ładowanie skryptów strony.  nowość  Oferty pracy  Pобота  Profile pracodawców  Moja kariera  Moje konto  Dla firmDodaj ogłoszenie  Oferta pracy  Start  Szukaj  Strefa ofert  Konto  Menu  eCRM Intern  5 BONSAI SPÓŁKA Z OGRANICZONĄ ODPOWIEDZIALNOŚCIĄO firmie  4\xa0500–5\xa0500  zł  brutto / mies.  Konstruktorska 12, Mokotów, WarszawaWarszawa, mazowieckie  ważna jeszcze 18 dnido 06 lipca 2024  umowa zlecenie  pełny etat, część etatu  praktykant / stażysta  praca hybrydowa  Praca od zaraz  Szukamy wielu kandydatówwakaty: 2  rekrutacja zdalna  System wynagrodzeń:stała podstawa wynagrodzenia  Specjalizacje:AI/ML  5 BONSAI SPÓŁKA Z OGRANICZONĄ ODPOWIEDZIALNOŚCIĄ  Konstruktorska 12  Mokotów  Warszawa  Sprawdź jak dojechać  Technologie, których używamy  Wymagane  Microsoft Excel  Mile widziane  JavaScript  HTML  CSS  System operacyjny      O projekcie  Interesujesz się rozwojem AI, nowinkami technologicznymi i digital marketingiem? Stawiasz pierwsze kroki w karierze zawodowej i chcesz mądrze wybrać kierunek? Mamy dla Ciebie propozycję. ;-)    Program Rozwojowy w dziale eCRM zakłada, że w ciągu ok 2 lat możesz stać się ekspertem w dynamicznie rozwijającej się branży automatyzacji marketingu i procesów biznesowych. Na początek jednak zapraszamy Cię na płatny staż. Poznajmy się bliżej!  Twój zakres obowiązków  Praca na bazach użytkowników, przygotowywanie segmentów i innych komponentów kampanii  Realizacja kampanii kanałami email, SMS / MMS, web push, mobile push itd.  Monitorowanie efektywności ścieżek automatyzacji, kampanii i rozwiązań wdrożonych na stronach www, prowadzenie testów A/B  Przygotowywanie wizualizacji danych i raportów w formie tabel lub prezentacji  Stała współpraca z zespołem Account Managerów i specjalistów eCRM  Nasze wymagania  Chęci szybkiego uczenia się, proaktywnej i otwartej postawy  Umiejętności analitycznego myślenia, formułowania wniosków, skrupulatności w pracy z kampaniami i liczbami  Dobrej organizacji pracy własnej, umiejętności priorytetyzacji zadań  Znajomości angielskiego minimum na poziomie C1  Zaawansowanej znajomości pakietu MS Office (PowerPoint, Excel)  Mile widziane  Posiadasz praktyczną znajomość SQLa i znasz podstawy JS, Pythona, HTML / CSS  Masz doświadczenie w pracy na oprogramowaniu typu SaaS lub na narzędziach do zarządzania treściami (CMS)  Jesteś studentem kierunków ścisłych jak matematyka, ekonometria, Big Data, MIESI lub związanych z marketingiem, e-biznesem  Tak organizujemy naszą pracę  Tak pracujemy  wewnątrz organizacji  rozwijasz kilka projektów jednocześnie  możesz zmienić projekt  masz wpływ na rozwiązania technologiczne  Skład zespołu  frontend developer  project manager  graphic designer  analityk biznesowy  Takie dajemy możliwości rozwoju  konferencje w Polsce  mentoring  szkolenia wewnątrzfirmowe  wymiana wiedzy technicznej w firmie  To oferujemy  Możliwość pracy hybrydowej (3 dni w biurze, 2 dni zdalnie)  Możliwość rozwoju w obszarze automatyzacji, omnichannel, strategii, analityki  Możliwość zdobycia certyfikatów poświadczających umiejętności  Fantastyczna atmosfera, imprezy integracyjne i pełne wsparcie w onboardingu, realizacji zadań i rozwoju ze strony przełożonych i całego zespołu  Benefity: Multisport, opiekę medyczną, dofinansowanie kursów językowych (po 3 miesiącach)  Benefity  dofinansowanie zajęć sportowych  prywatna opieka medyczna  dofinansowanie nauki języków  owoce  spotkania integracyjne  brak dress code’u  kawa / herbata  5 BONSAI SPÓŁKA Z OGRANICZONĄ ODPOWIEDZIALNOŚCIĄ  Przewiń do profilu firmy  5BONSAI to agencja CRM & Automatyzacji Marketingu z siedzibą w Warszawie, która wykorzystuje rozwiązania oparte na sztucznej inteligencji, aby pomagać firmom budować silniejsze relacje z klientami we wszystkich punktach styku. Z 15-letnim doświadczeniem w marketingu cyfrowym na rynku europejskim oraz współpracą z globalnymi markami, 5BONSAI to zespół ponad 50 Ekspertów eCRM i specjalistów MarTech. Jako partnerzy czołowych dostawców, takich jak Salesforce i Synerise, świadczymy kompleksowe usługi CRM, Automatyzacji Marketingu oraz Marketingu Lojalnościowego, skupiając się na wielokanałowych doświadczeniach, hiperpersonalizacji, strategiach opartych na danych oraz rozwiązaniach opartych na sztucznej inteligencji do prognoz i trendów.  Aplikuj szybko  Aplikuj szybko  Zapisz  Drukuj  Udostępnij  co to?  Praca  Warszawa  Internet / e-Commerce / Nowe media  E-marketing / SEM / SEO  Aplikuj szybko  Zapisz  Drukuj  Udostępnij  co to?  eCRM Intern, Konstruktorska 12, Mokotów, Warszawa  Dla kandydatów  Pomoc  Pracuj w Grupie Pracuj  Festiwal Pracy JOBICON  Kalkulator godzinowy  Dla firm  Dodaj ogłoszenie  Konto pracodawcy  Pomoc dla firm  Porady dla firm  Porady rekrutacyjne  Grupa Pracuj  O nas  Centrum Prasowe  Reklama  Partnerzy  Archiwum ofert  Narzędzia  Kreator CV  Porady zawodowe  Kalkulator zarobków  Zarobki  The:blog  Nasze produkty  Pobierz aplikację  © Grupa Pracuj S.A.  Regulamin  Polityka Prywatności  Polityka plików cookies  Akt o usługach cyfrowych",
        JobOffer(
            position_name="eCRM Intern",
            company_name="5 BONSAI",
            position_level="praktykant / stażysta",
            specializations=[
                "AI",
                "ML",
            ],
            skills=[
                "Microsoft Excel",
                "JavaScript",
                "HTML",
                "CSS",
                "Python",
                "SQL",
            ],
            requirements=[
                "Higher technical education (student of 3-6 years of study or graduates)",
                "Programming and software development skills (experience with Python)",
                "Solid background in mathematics and statistics",
                "Good skills in relational database structure and operations (MySQL, Microsoft SQL)",
                "Knowledge in time-series forecasting, statistical analysis, Machine Learning, deep learning, econometrics, text processing",
                "English at Intermediate level and higher",
            ],
            responsibilities=[
                "Participation in the development of own product based on Machine Learning",
                "Development and implementation of generative AI products",
                "Driving data science projects and solutions with domain expert consultants",
                "Building the team, systems, and practices that enable SMART business to leverage cutting-edge data mining and machine learning techniques",
                "Building, testing, and validating predictive models",
                "Cooperation with our Software and Data Engineers according to requirements",
                "Modifications of the implemented solutions",
                "Performance reporting",
            ],
            about_position="SMART business, an international IT company with offices in Ukraine, Georgia, Azerbaijan, and Poland, specializes in ERP, CRM, and HRM systems using Microsoft services. They offer paid internships for Junior Microsoft Dynamics 365 Developers to students and graduates in technical fields. Interns undergo training and project implementation. The company values collaboration, excellence, change, resilience, and transformation. To apply, candidates must complete a test assignment.",
        ),
    ),
]


class ExtractingAgent:
    def __init__(
        self, model: str = "gpt-3.5-turbo", temperature: float = 0, api_key=""
    ):
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                You are an expert extraction algorithm.
                Extract only relevant details from the text:
                - position_name: The name of the position.
                - company_name: The name of the company offering the job.
                - position_level: The level or seniority of the position.
                - specializations: The specializations required for the position.
                - skills: The skills required for the position.
                - about_position: A brief description about the position.
                - responsibilities: The responsibilities associated with the position.
                - requirements: The requirements needed to qualify for the position.
                - about_company: A brief description about the company offering the job.
                If there are additional details do not include them.
                Translate value of attribute always to English.
                If you do not know the value of an attribute asked to extract,
                return null for the attribute's value.
                """,
                ),
                MessagesPlaceholder("examples"),
                ("human", "{text}"),
            ]
        )
        #         self.llm = OllamaFunctions(
        #             model="phi3:medium",
        #             format="json",
        #             temperature=temperature,
        #             keep_alive=0,
        #         )
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )
        self.runnable = self.prompt | self.llm.with_structured_output(schema=JobOffer)

    def extract_job_offer(self, html_data: str, messages):
        print("[INFO] Extracting Agent working")
        response = self.runnable.invoke({"text": html_data, "examples": messages})
        print("[INFO] Extracting Agent done working")
        return response

    def create_directories(self, response: JobOffer):
        if not os.path.exists("./data"):
            os.mkdir("./data")

        position_name = response.position_name.replace("/", "_")
        company_name = response.company_name.replace("/", "_")

        path = f"./data/{company_name}/{position_name}/"
        os.makedirs(path, exist_ok=True)
        print(f"[INFO] Created directory {path}")

        return path

    def save_job_listing(self, html_data: str, path: str):
        path = os.path.join(path, "job_listing.txt")
        with open(path, "w") as file:
            file.write(html_data)
        print(f"[INFO] Saved {path}")

    def save_job_information(self, response: JobOffer, url: str, path: str):
        response_data = response.dict()
        response_data["url"] = url

        json_path = os.path.join(path, "job_information.json")
        with open(json_path, "w") as json_file:
            json.dump(response_data, json_file, indent=4)
        print(f"[INFO] Saved {json_path}")

    def url_data(self, url: str, html_path: str = "./job_listing.html") -> str:
        print(f"[INFO] Request")
        response = requests.get(url)
        print(f"[INFO] Response")
        html_content = response.text
        with open(html_path, "w", encoding="utf-8") as file:
            file.write(html_content)
        print(f"[INFO] HTML SAVED")
        loader = UnstructuredHTMLLoader(html_path)
        html_data = loader.load()
        html_data = html_data[0].page_content
        html_data = html_data.replace("\n", " ")
        return html_data

    def process_job_offer(self, url: str):
        html_data = self.url_data(url)
        messages = create_example_messages(examples)
        response = self.extract_job_offer(html_data, messages)
        path = self.create_directories(response)
        self.save_job_listing(html_data, path)
        self.save_job_information(response, url, path)
        print(f"JSON data saved to {os.path.join(path, 'job_information.json')}")


if __name__ == "__main__":
    extractor = ExtractingAgent(api_key=extract_secret("OPENAI_API_KEY"))
    url = "https://www.pracuj.pl/praca/ai-ml-engineer-warszawa,oferta,1003363060?s=4921b6b8&searchId=MTcxODgyNjgwMTcwOC4zOTk3"
    extractor.process_job_offer(url)
