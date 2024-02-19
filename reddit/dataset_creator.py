"""
Script to create a dataset from the Change My View (CMV) subreddit.
"""

import json
import random
import time
import re
import pandas as pd

# Configure paths
RAW_DATA_PATH = "../data/reddit/cmv_20161111"
SAVE_PATH_CSV = "../data/reddit/reddit.csv"
SAVE_PATH_JSON = "../data/reddit/reddit.json"


class Post:
    """
    Class to represent a Reddit post.

    Args:
        raw_post (dict): Raw post data

    Attributes:
        author (str): Author of the post
        body (str): Body of the post
        id (str): ID of the post
        op_delta_comment (bool): Whether the post has a delta comment made by OP
        comments (list): List of comments in the post

    Methods:
        assign_delta: Assigns delta to the comments within the post

    """

    def __init__(self, raw_post: dict) -> None:
        self.author = raw_post["author"]
        self.body = raw_post["selftext"]
        self.id = raw_post["name"]
        self.op_delta_comment = None
        self.comments = [
            Comment(raw_comment, raw_post["comments"])
            for raw_comment in raw_post["comments"]
            if raw_comment["parent_id"] == self.id
        ]

    def assign_delta(self) -> None:
        """
        Assigns delta to the comments within the post.

        Returns:
            None
        """
        self.op_delta_comment = False
        for comment in self.comments:
            comment.assign_delta(self)

    def __str__(self) -> str:
        return f"{self.id} ({self.author})"

    def __repr__(self) -> str:
        return f"{self.id} ({self.author})"


class Comment:
    """
    Class to represent a Reddit comment.

    Args:
        raw_comment (dict): Raw comment data
        raw_comments (list): List of raw comments with the same post

    Attributes:
        author (str): Author of the comment
        body (str): Body of the comment
        id (str): ID of the comment
        delta (bool): Whether the comment has a delta
        op_delta (bool): Whether the comment has a delta made by the OP
        deltabot_confirmation (bool): Whether the comment is a deltabot delta confirmation
        children (list): List of responses to the comment

    Methods:
        assign_delta: Assigns delta to the comment
    """

    def __init__(self, raw_comment: dict, raw_comments: list) -> None:
        self.author = raw_comment["author"] if "author" in raw_comment else None
        self.body = raw_comment["body"] if "body" in raw_comment else None
        self.id = raw_comment["name"] if "name" in raw_comment else None
        self.delta = None
        self.op_delta = None
        self.deltabot_confirmation = (
            True
            if (
                self.author == "DeltaBot"
                and (
                    self.body.startswith("Confirmed: 1")
                    or self.body.startswith("Confirmed - 1")
                    or self.body.startswith("Confirmed -- 1")
                )
            )
            else False
        )
        self.children = [
            Comment(raw_comment, raw_comments)
            for raw_comment in raw_comments
            if raw_comment["parent_id"] == self.id
        ]

    def assign_delta(self, post: Post):
        """
        Assigns delta to the comment.

        Args:
            post (Post): Post containing the comment

        Returns:
            None
        """
        self.delta, self.op_delta = False, False
        for child in self.children:
            for grandchild in child.children:
                if grandchild.deltabot_confirmation:
                    self.delta = True
                    self.op_delta = bool(
                        (child.author == post.author) and (self.author != post.author)
                    )
                    post.op_delta_comment = self.op_delta
                    if self.op_delta:
                        return

    def __str__(self) -> str:
        return f"{self.id} ({self.author})"

    def __repr__(self) -> str:
        return f"{self.id} ({self.author})"


class DatasetCreator:
    """
    Class to create a dataset from the Change My View (CMV) subreddit.

    Args:
        path (str): Path to the raw data file

    Attributes:
        posts (list): List of posts in the dataset
        op_delta_posts (list): List of posts with OP delta comments
        op_delta_comments (list): List of OP delta comments
        no_op_delta_comments (list): List of comments without OP delta
        selected_no_op_delta_comments (list): List of selected comments without OP delta
        selected_comments (list): List of selected comments

    Methods:
        read_posts: Reads the raw data file and creates a list of posts
        assign_deltas: Assigns deltas to the comments within the posts
        get_op_delta_posts: Returns a list of posts with OP delta comments
        get_op_delta_comments: Returns a list of OP delta comments
        get_no_op_delta_comments: Returns a list of comments without OP delta
        get_selected_no_op_delta_comments: Returns a list of selected comments without OP delta
        get_selected_comments: Returns a list of selected comments
        clean_markdown: Cleans markdown from the text
        clean_selected_comments: Cleans the selected comments
        save_to_csv: Saves the selected comments to a CSV file
        save_to_json: Saves the selected comments to a JSON file
    """

    def __init__(self, path) -> None:
        self.posts = self.read_posts(path)

    def extract(self) -> None:
        """
        Sample comments for the dataset.

        Returns:
            None
        """
        self.assign_deltas()
        self.op_delta_posts = self.get_op_delta_posts()
        self.op_delta_comments = self.get_op_delta_comments()
        self.no_op_delta_comments = self.get_no_op_delta_comments()
        self.selected_no_op_delta_comments = self.get_selected_no_op_delta_comments()
        self.selected_comments = self.get_selected_comments()

    def read_posts(self, path: str) -> list:
        """
        Reads the raw data file and creates a list of posts.

        Args:
            path (str): Path to the raw data file

        Returns:
            list: List of posts in the dataset
        """
        with open(path, "r", encoding="utf8") as file:
            raw_posts = [json.loads(line) for line in file]
        return [Post(post) for post in raw_posts]

    def assign_deltas(self) -> None:
        """
        Assigns deltas to the comments within the posts.

        Returns:
            None
        """
        for post in self.posts:
            post.assign_delta()

    def get_op_delta_posts(self) -> list:
        """
        Returns a list of posts with OP delta comments.

        Returns:
            list: List of posts with OP delta comments
        """
        return [post for post in self.posts if post.op_delta_comment]

    def get_op_delta_comments(self) -> list:
        """
        Returns a list of OP delta comments.

        Returns:
            list: List of OP delta comments
        """
        return [
            comment
            for post in self.op_delta_posts
            for comment in post.comments
            if comment.op_delta
        ]

    def get_no_op_delta_comments(self) -> list:
        """
        Returns a list of comments without OP delta from the op_delta_posts.

        Returns:
            list: List of comments without OP delta from the op_delta_posts
        """
        return [
            comment
            for post in self.op_delta_posts
            for comment in post.comments
            if not comment.op_delta
        ]

    def get_selected_no_op_delta_comments(self, random_seed: int = 42) -> list:
        """
        Returns a list of selected comments without OP delta.

        Args:
            random_seed (int): Random seed for reproducibility

        Returns:
            list: List of selected comments without OP delta
        """
        random.seed(random_seed)
        return random.sample(self.no_op_delta_comments, len(self.op_delta_comments))

    def get_selected_comments(self, random_seed: int = 42) -> list:
        """
        Returns a list of selected comments.

        Args:
            random_seed (int): Random seed for reproducibility

        Returns:
            list: List of selected comments
        """
        random.seed(random_seed)
        selected_comments = self.op_delta_comments + self.selected_no_op_delta_comments
        random.shuffle(selected_comments)
        return selected_comments

    def clean_markdown(self, text: str) -> str:
        """
        Cleans markdown from the text.

        Args:
            text (str): Text to be cleaned

        Returns:
            str: Cleaned text
        """
        text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)
        text = re.sub(r"_{1,2}(.*?)_{1,2}", r"\1", text)
        text = re.sub(r"`(.*?)`", r"\1", text)
        text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
        text = text.replace("\n", " ")
        return text

    def clean_selected_comments(self) -> None:
        """
        Cleans the selected comments.

        Returns:
            None
        """
        for comment in self.selected_comments:
            comment.body = self.clean_markdown(comment.body)

    def save_to_csv(self, path: str) -> None:
        """
        Saves the selected comments to a CSV file.

        Args:
            path (str): Path to save the CSV file

        Returns:
            None
        """
        pd.DataFrame(
            [
                {"text": comment.body, "delta": comment.op_delta}
                for comment in self.selected_comments
            ]
        ).to_csv(path, index=False)

    def save_to_json(self, path: str) -> None:
        """
        Saves the selected comments to a JSON file.

        Args:
            path (str): Path to save the JSON file

        Returns:
            None
        """
        with open(path, "w", encoding="utf8") as file:
            json.dump(
                [
                    {"text": comment.body, "delta": comment.op_delta}
                    for comment in self.selected_comments
                ],
                file,
                indent=4,
            )

    def __str__(self) -> str:
        return f"DatasetCreator: {len(self.posts)} posts"


def main():
    start_time = time.time()
    print("Creating DatasetCreator...")
    dataset_creator = DatasetCreator(RAW_DATA_PATH)
    print("Extracing Comments...")
    dataset_creator.extract()
    print("Saving Raw...")
    dataset_creator.save_to_csv(SAVE_PATH_CSV)
    dataset_creator.save_to_json(SAVE_PATH_JSON)
    print("Cleaning...")
    dataset_creator.clean_selected_comments()
    print("Saving Cleaned...")
    dataset_creator.save_to_csv(SAVE_PATH_CSV.replace(".csv", "_cleaned.csv"))
    dataset_creator.save_to_json(SAVE_PATH_JSON.replace(".json", "_cleaned.json"))
    print("Done!")
    end_time = time.time()
    print(f"Processed finished in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
