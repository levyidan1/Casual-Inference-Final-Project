"""
This file is used to scrape data from the MMRF-COMPASS website.
Specifically, it goes over all the patients, and for each patient, it goes to the Follow-Ups tab,
and then it goes over all the follow-ups, and for each follow-up, it saves the data in a CSV file.
"""
from typing import List

import pandas as pd
import selenium
import selenium.common.exceptions
import selenium.webdriver
import selenium.webdriver.common.by
import selenium.webdriver.common.keys
import selenium.webdriver.support.expected_conditions
import selenium.webdriver.support.ui
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager


def get_driver() -> selenium.webdriver:
    """Get a Selenium driver."""
    driver = selenium.webdriver.Chrome(ChromeDriverManager().install())
    driver.implicitly_wait(10)
    return driver


def click_on_accept_button(driver: selenium.webdriver) -> None:
    """Click on the "Accept" button."""
    button = driver.find_element_by_css_selector('button[data-test="modal-cancel-button"]')
    button.click()


def click_on_follow_ups_tab(driver: selenium.webdriver) -> None:
    """Click on the "Follow-Ups" tab."""
    follow_up_tab = driver.find_element_by_css_selector(
        "#clinical > div > div > div.test-tabs > div:nth-child(1) > div:nth-child(5) > p")
    follow_up_tab.click()


def get_soup(driver: selenium.webdriver) -> BeautifulSoup:
    """Get the BeautifulSoup object from the current page."""
    return BeautifulSoup(driver.page_source, "html.parser")


def get_case_ids() -> List[str]:
    """Get the case IDs from clinical_data/clinical.tsv file."""
    df = pd.read_csv("../clinical_data/clinical.tsv", sep="\t")
    return list(set(df["case_id"].tolist()))


def get_follow_ups(driver: selenium.webdriver) -> List[str]:
    """Get the follow-ups from the current page."""
    follow_ups_elements = driver.find_elements_by_css_selector(
        '#clinical > div > div > div.test-tabs > div:nth-child(2) > div > div > div > div:nth-child(1) > div > div')
    return [element.text for element in follow_ups_elements]


def extract_follow_up_data(driver: selenium.webdriver) -> pd.DataFrame:
    """Extract the follow-up data from the current page."""
    soup = get_soup(driver)
    table = soup.select_one(
        "#clinical > div > div > div.test-tabs > div:nth-child(2) > div > div > div > div:nth-child(2) > div:nth-child(3) > table")
    if not table:
        return pd.DataFrame()
    data = {}
    headers = [th.text for th in table.find("thead").find_all("th")]
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        for i, td in enumerate(tds):
            data.setdefault(headers[i], []).append(td.text)
    return pd.DataFrame(data)


def get_follow_up_days(driver: selenium.webdriver) -> int:
    """Get the follow-up days from the current page."""
    selector = "#clinical > div > div > div.test-tabs > div:nth-child(2) > div > div > div > div:nth-child(2) > div.undefined.test-entity-table-wrapper > table > tbody > tr:nth-child(2) > td"
    try:
        return int(driver.find_element_by_css_selector(selector).text)
    except selenium.common.exceptions.NoSuchElementException as e:
        print(f"No follow up days found: {e}")
        return 0
    except ValueError as e:
        print(f"Follow up days not integers: {e}")
        return 0


def drop_irrelevant_data(follow_up_data: pd.DataFrame) -> pd.DataFrame:
    """Drop irrelevant data from the follow-up data."""
    columns_to_drop = ["UUID", "Gene Symbol", "Second Gene Symbol", "Molecular Analysis Method", "Test Result",
                       "Biospecimen Type", "Variant Type", "Chromosome", "AA Change", "Antigen",
                       "Mismatch Repair Mutation"]
    coloumns_to_drop = [x for x in columns_to_drop if x in follow_up_data.columns]
    follow_up_data = follow_up_data.drop(coloumns_to_drop, axis=1)
    return follow_up_data


def reorder_data(case_ids_data: pd.DataFrame) -> pd.DataFrame:
    """Reorder the columns in the data."""
    case_ids_data = case_ids_data.sort_values(["Case ID", "Days to Follow-Up"], ascending=[True, True])
    columns = ["Case ID", "Follow-Up", "Days to Follow-Up", "Laboratory Test", "Test Value", "Test Units",
               "Patient Height", "Patient Weight"]
    case_ids_data["Test Value"] = case_ids_data["Test Value"].str.extract(r"(\d+\.?\d*)", expand=False).astype(float)
    return case_ids_data[columns]


def get_patient_height(driver: selenium.webdriver) -> int:
    """Get the patient height from the current page."""
    selector = "#clinical > div > div > div.test-tabs > div:nth-child(2) > div > div > div > div:nth-child(2) > div.undefined.test-entity-table-wrapper > table > tbody > tr:nth-child(9) > td"
    try:
        return int(driver.find_element_by_css_selector(selector).text)
    except selenium.common.exceptions.NoSuchElementException as e:
        print(f"No patient height found: {e}")
        return 0
    except ValueError as e:
        print(f"Patient height not integers: {e}")
        return 0


def get_patient_weight(driver: selenium.webdriver) -> int:
    """Get the patient weight from the current page."""
    selector = "#clinical > div > div > div.test-tabs > div:nth-child(2) > div > div > div > div:nth-child(2) > div.undefined.test-entity-table-wrapper > table > tbody > tr:nth-child(10) > td"
    try:
        return int(driver.find_element_by_css_selector(selector).text)
    except selenium.common.exceptions.NoSuchElementException as e:
        print(f"No patient weight found: {e}")
        return 0
    except ValueError as e:
        print(f"Patient weight not integers: {e}")
        return 0


def scrape_case_id(case_id: str, driver: selenium.webdriver) -> pd.DataFrame:
    """Scrape the follow-ups for the given case ID."""
    follow_ups = get_follow_ups(driver)
    follow_ups_data = pd.DataFrame()
    follow_up_elements = driver.find_elements_by_css_selector(
        "#clinical > div > div > div.test-tabs > div:nth-child(2) > div > div > div > div:nth-child(1) > div > div")
    patient_height = 0
    patient_weight = 0
    for i, follow_up_element in enumerate(follow_up_elements):
        follow_up_element.click()
        patient_height, patient_weight = max(patient_height, get_patient_height(driver)), max(patient_weight,
                                                                                              get_patient_weight(
                                                                                                  driver))
        follow_up_data = extract_follow_up_data(driver)
        follow_up_data["Follow-Up"] = follow_ups[i]
        follow_up_data["Days to Follow-Up"] = get_follow_up_days(driver)
        follow_up_data = drop_irrelevant_data(follow_up_data)
        follow_ups_data = follow_ups_data.append(follow_up_data, ignore_index=True)
    follow_ups_data["Patient Height"] = patient_height
    follow_ups_data["Patient Weight"] = patient_weight
    follow_ups_data["Case ID"] = case_id
    return follow_ups_data


def scrape_case_ids(case_ids: List[str]) -> None:
    """Scrape the follow-ups for the given case IDs."""
    driver = get_driver()
    case_ids_data = pd.DataFrame()
    for i, case_id in enumerate(case_ids):
        driver.get(f"https://portal.gdc.cancer.gov/cases/{case_id}")
        if i == 0:
            click_on_accept_button(driver)
        click_on_follow_ups_tab(driver)
        case_ids_data = case_ids_data.append(scrape_case_id(case_id, driver), ignore_index=True)
    driver.quit()
    case_ids_data = reorder_data(case_ids_data)
    case_ids_data.to_csv("follow_ups_data.csv", index=False)


if __name__ == "__main__":
    case_ids = get_case_ids()
    scrape_case_ids(case_ids)
